import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import PIL
import numpy as np
#no marugoto dependency
import re
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
import json
import h5py
import uni
import os
from collections.abc import Iterable
import timm
import time
import logging
import cv2
from PIL import Image
import itertools
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from conch.open_clip_custom import create_model_from_pretrained
from typing import TypeVar, Callable
import sys
import utils
from datetime import timedelta
from typing import Any, Optional, Callable, Tuple
from fastai.vision.all import Path, get_image_files, verify_images
from swin_transformer import swin_tiny_patch4_window7_224, ConvStem

__version__ = "001_01-10-2023"

def get_digest(file: str):
    sha256 = hashlib.sha256()
    with open(file, 'rb') as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

class FeatureExtractorCTP:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles.
        """
        digest = get_digest(self.checkpoint_path)
        assert digest == '7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539'

        self.model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        self.model.head = nn.Identity()

        ctranspath = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ctranspath['model'], strict=True)
        
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model_name='xiyuewang-ctranspath-7c998680'

        print("CTransPath model successfully initialised...\n")
        return model_name
        
class FeatureExtractorChiefCTP:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles.
        """
        #Architecture is identical, only the weights differ from CTP
        self.model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        self.model.head = nn.Identity()

        ctranspath = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ctranspath['model'], strict=True)
        
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model_name='chief-ctp'

        print("ChiefCTP model successfully initialised...\n")
        return model_name
    
class FeatureExtractorUNI:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles. 
        Requirements: 
            Permission from authors via huggingface: https://huggingface.co/MahmoodLab/UNI
            Huggingface account with valid login token
        On first model initialization, you will be prompted to enter your login token. The token is
        then stored in ./home/<user>/.cache/huggingface/token. Subsequent inits do not require you to re-enter the token. 

        Args:
            device: "cuda" or "cpu"
        """
        asset_dir = f"{os.environ['STAMP_RESOURCES_DIR']}/uni"
        model, transform = uni.get_encoder(enc_name="uni", device=device, assets_dir=asset_dir)
        self.model = model
        self.transform = transform

        digest = get_digest(f"{asset_dir}/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin")
        model_name = f"mahmood-uni-{digest[:8]}"

        print("UNI model successfully initialised...\n")
        return model_name
    

class FeatureExtractorProvGP:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using GigaPath tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'prov-gigapath'
        checkpoint = 'pytorch_model.bin'
        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

        # Ensure the directory exists
        os.makedirs(ckpt_dir, exist_ok=True)

        # Load the model structure
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)

        # Load the state dict from the checkpoint file
        self.model.load_state_dict(torch.load(ckpt_path))

        # If a CUDA device is available, move the model to the device
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        # Define the transform
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        print("GigaPath tile encoder model successfully initialized...\n")
        return model_name

class FeatureExtractorHibouB:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using Hibou B tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'hibou-b'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        processor_path = os.path.join(ckpt_dir, 'processor')

        # Load the processor
        self.processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

        # Load the model configuration
        config = AutoConfig.from_pretrained("histai/hibou-b", trust_remote_code=True)

        # Initialize the model with the config
        self.model = AutoModel.from_config(config, trust_remote_code=True)

        # Load the saved model weights
        self.model.load_state_dict(torch.load(ckpt_path))

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7068, 0.5755, 0.7220], std=[0.1950, 0.2316, 0.1816])
        ])

        print("Hibou-B model successfully initialised...\n")
        return model_name

class FeatureExtractorHibouL:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using Hibou L tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'hibou-l'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        processor_path = os.path.join(ckpt_dir, 'processor')

        # Load the processor
        self.processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

        # Load the model configuration
        config = AutoConfig.from_pretrained("histai/hibou-L", trust_remote_code=True)

        # Initialize the model with the config
        self.model = AutoModel.from_config(config, trust_remote_code=True)

        # Load the saved model weights
        self.model.load_state_dict(torch.load(ckpt_path))

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7068, 0.5755, 0.7220], std=[0.1950, 0.2316, 0.1816])
        ])

        print("Hibou-L model successfully initialised...\n")
        return model_name

class FeatureExtractorKaiko:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using Kaiko tile encoder."""
        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"

        model_name = 'kaiko-vitl14'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        
        self.model = torch.hub.load("kaiko-ai/towards_large_pathology_fms", "vitl14", trust_repo=True, pretrained=False)
        self.model.load_state_dict(torch.load(ckpt_path))

        from torchvision.transforms import v2
        # initialize the model pre-process transforms
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=224),
                v2.CenterCrop(size=224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

        print("Kaiko model successfully initialised...\n")
        return model_name

class FeatureExtractorConch:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using CONCH tile encoder."""
        
        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'conch'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)

        # Initialize the model (you may need to pass other necessary parameters as per the create_model_from_pretrained function)
        self.model, self.processor = create_model_from_pretrained('conch_ViT-B-16', ckpt_path, force_image_size=224)

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        r"""OpenAI color normalization mean in RGB format (values in 0-1)."""
        r"""OpenAI color normalization std in RGB format (values in 0-1)."""
        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        r"""ImageNet color normalization mean in RGB format (values in 0-1)."""
        r"""ImageNet color normalization std in RGB format (values in 0-1)."""
        IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)

        from typing import Sequence
        mean: Sequence[float] = OPENAI_DATASET_MEAN
        std: Sequence[float] = OPENAI_DATASET_STD

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        print("CONCH model successfully initialised...\n")
        return model_name

class FeatureExtractorPhikon:
    def init_feat_extractor(self, device: str, weights_path: str = None, **kwargs):
        """Extracts features from slide tiles using Phikon tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'phikon'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        processor_path = os.path.join(ckpt_dir, 'processor')

        # Load the processor
        self.processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

        # Load the model configuration
        config = AutoConfig.from_pretrained("owkin/phikon", trust_remote_code=True)

        # Initialize the model with the config
        self.model = AutoModel.from_config(config, trust_remote_code=True)

        # Load the saved model weights
        self.model.load_state_dict(torch.load(ckpt_path))

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Phikon model successfully initialized...\n")
        return model_name

class FeatureExtractorVirchow:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using Virchow tile encoder."""

        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'virchow'
        checkpoint = 'pytorch_model.bin'
        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

        # Ensure the directory exists
        os.makedirs(ckpt_dir, exist_ok=True)

        # Load the model structure
        self.model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=False, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

        # Load the state dict from the checkpoint file
        self.model.load_state_dict(torch.load(ckpt_path))

        # If a CUDA device is available, move the model to the device
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        # Define the transform
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        
        print("Virchow model successfully initialized...\n")
        return model_name
    
class FeatureExtractorVirchow2:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using Virchow2 tile encoder."""

        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers import SwiGLUPacked

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'virchow2'
        checkpoint = 'pytorch_model.bin'
        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

        # Ensure the directory exists
        os.makedirs(ckpt_dir, exist_ok=True)

        # Load the model structure
        self.model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=False, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

        # Load the state dict from the checkpoint file
        self.model.load_state_dict(torch.load(ckpt_path))

        # If a CUDA device is available, move the model to the device
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        # Define the transform
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        
        print("Virchow2 model successfully initialized...\n")
        return model_name

class FeatureExtractorHOptimus0:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using H-optimus-0 tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'hoptimus0'
        checkpoint = 'pytorch_model.bin'
        ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

        self.model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=False, init_values=1e-5, dynamic_img_size=False)
        
        self.model.load_state_dict(torch.load(ckpt_path))

        # If a CUDA device is available, move the model to the device
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])

        print("H-optimus-0 model successfully initialized...\n")
        return model_name

class FeatureExtractorPLIP:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using PLIP tile encoder."""
        
        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'plip'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        processor_path = os.path.join(ckpt_dir, 'processor')

        # Load the processor
        self.processor = AutoImageProcessor.from_pretrained(processor_path, trust_remote_code=True)

        # Load the model configuration
        config = AutoConfig.from_pretrained("vinid/plip", trust_remote_code=True)

        # Initialize the model with the config
        self.model = AutoModel.from_config(config, trust_remote_code=True)

        self.model.load_state_dict(torch.load(ckpt_path))

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print("PLIP model successfully initialised...\n")
        return model_name

class FeatureExtractorBiomedCLIP:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using BiomedCLIP tile encoder."""
        
        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'biomedclip'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)

        from open_clip import create_model_from_pretrained

        self.model, self.processor = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', ckpt_path)

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print("BiomedCLIP model successfully initialised...\n")
        return model_name
    
class FeatureExtractorDinoSSLPath:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using DinoSSLPath tile encoder."""
        from timm.models.vision_transformer import VisionTransformer
        
        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'dinosslpath'
        checkpoint = 'pytorch_model.bin'

        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)

        self.model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=384, num_heads=6, num_classes=0
        )

        self.model.load_state_dict(torch.load(ckpt_path))

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        LUNIT_MEAN = (0.70322989, 0.53606487, 0.66096631)
        LUNIT_STD = (0.21716536, 0.26081574, 0.20723464)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=LUNIT_MEAN, std=LUNIT_STD)
        ])

        print("DinoSSLPath model successfully initialised...\n")
        return model_name    

T = TypeVar("T")

class SlideTileDataset(Dataset[T]):
    def __init__(self,
                 root: str,
                 verify_image: bool = False,
                 transform: Optional[Callable] = None,
                 ) -> None:

        super().__init__()
    
        self.root = Path(root).expanduser()
        self.verify_image = verify_image
        self.transform = transform
        self.image_paths = get_image_files(self.root)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> T:
        try:
            image_path = self.image_paths[index]
            image = Image.open(image_path).convert(mode="RGB")
            image = self.transform(image)
            return image
        except:
            return self[index+1]

def batched(iterable: Iterable[T], batch_size: int) -> Iterable[list[T]]:
    # batched('ABCDEFG', 3) → ABC DEF G
    if batch_size < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def extract_features_(
        *,
        model: nn.Module, model_name, transform: Callable[[PIL.Image.Image], torch.Tensor],
        outdir: Path, slide_url:Path,
        augmented_repetitions: int = 0, cores: int = 8, is_norm: bool = True, device: str = 'cpu',
        target_microns: int = 256, patch_size: int = 224, processor=None
) -> None:
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """

    extractor_string = f'STAMP-extract-{__version__}_{model_name}'
    with open(outdir.parent/'info.json', 'w') as f:
        json.dump({'extractor': extractor_string,
                  'augmented_repetitions': augmented_repetitions,
                  'normalized': is_norm,
                  'microns': target_microns,
                  'patch_size': patch_size}, f)
    
    unaugmented_ds: Dataset[PIL.Image.Image] = SlideTileDataset(slide_url, transform=lambda x: x)
    augmented_ds = []

    coords = [[int(x), int(y)] for f in os.listdir(slide_url) if (m := re.search(r'(\d+)x_(\d+)y', f)) for x, y in [m.groups()]]
    #clean up memory

    ds: Dataset[PIL.Image.Image] = ConcatDataset([unaugmented_ds, augmented_ds])

    dl = torch.utils.data.DataLoader(
        ds, batch_size=None, shuffle=False, num_workers=cores, drop_last=False, pin_memory=(device != 'cpu'))


    # We do this because we can't put PIL images into a tensor
    # / would have to transform back and forth a buch of times.
    # FIXME: Rewrite all of this so we unify all the preprocessing
    # following e.g. `transformers.image_processing_utils.BaseImageProcessor`

    from math import ceil

    batch_size = 64

    batched_dl = batched(dl, batch_size=batch_size)

    batched_dl: Iterable[list[PIL.Image.Image]]

    model = model.eval().to(device)
    dtype = next(model.parameters()).dtype

    feats = []

    class_feats = []

    # Calculate the total number of batches
    total_batches = ceil(len(dl) / batch_size)

    with torch.inference_mode():
        for batch in tqdm(batched_dl, leave=False, total=total_batches):
            batch: list[PIL.Image.Image]

            if model_name == "hibou-b" or model_name == "hibou-l" or model_name == "phikon":
                # Ensure the batch is correctly normalized
                hf_data = processor(
                    images=batch,
                    return_tensors="pt"
                ).to(device)

                hf_output = model(**hf_data)
                output = hf_output.pooler_output # Use last_hidden_state for detailed spatial info
                feats.append(output.cpu().detach().half())
            elif model_name == "biomedclip":
                processed_images = []
                for image in batch:
                    processed = processor(image)
                    processed_images.append(processed)
                processed_batch = torch.stack(processed_images).to(device, dtype=dtype)
                logits = model.encode_image(processed_batch)
                feats.append(logits.cpu().detach().half())
            elif model_name == "plip":
                inputs = processor(images=batch, return_tensors="pt").to(device)
                output = model.get_image_features(**inputs)
                feats.append(output.cpu().detach().half())
            elif model_name == "conch":
                processed_images = []
                for image in batch:
                    processed = processor(image)
                    processed_images.append(processed)
                processed_batch = torch.stack(processed_images).to(device, dtype=dtype)
                output = model.encode_image(processed_batch, proj_contrast=False, normalize=False)
                feats.append(output.cpu().detach().half())

            elif model_name == "virchow" or model_name == "virchow2":
                processed_batch = torch.stack([transform(img) for img in batch]).to(device, dtype=dtype)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(processed_batch.type(dtype).to(device))

                class_token = output[:, 0]
                if model_name == "virchow":
                    patch_tokens = output[:, 1:]
                else:
                    patch_tokens = output[:, 5:]
                embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
                embedding = embedding.to(torch.float16)
                feats.append(embedding)
                class_token = class_token.to(torch.float16)
                class_feats.append(class_token)
            else:
                processed_batch = torch.stack([transform(img) for img in batch]).to(device, dtype=dtype)
                feats.append(
                    model(processed_batch.type(dtype).to(device)).half().cpu().detach())

        all_feats = torch.concat(feats)
        if model_name == "virchow" or model_name == "virchow2":
            all_class_feats = torch.concat(class_feats)

    with h5py.File(f'{outdir}.h5', 'w') as f:
        f['coords'] = coords
        if model_name == "mahmood-uni-56ef09b4":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 1024, all_feats.shape
        elif model_name == "xiyuewang-ctranspath-7c998680":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 768, all_feats.shape
        elif model_name == "chief-ctp":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 768, all_feats.shape
        elif model_name == "prov-gigapath":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 1536, all_feats.shape
        elif model_name == "hibou-b":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 768, all_feats.shape
        elif model_name == "hibou-l":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 1024, all_feats.shape
        elif model_name == "kaiko-vitl14":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 1024, all_feats.shape
        elif model_name == "conch":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 512, all_feats.shape
        elif model_name == "phikon":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 768, all_feats.shape
        elif model_name == "virchow" or model_name == "virchow2":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 2560, all_feats.shape
            assert len(all_class_feats.shape) == 2 and all_class_feats.shape[1] == 1280, all_feats.shape
            with h5py.File(f'{outdir}_class_tokens.h5', 'w') as g:
                g['coords'] = coords
                g['feats'] = all_class_feats.cpu().numpy()
                g['augmented'] = np.repeat([False, True], [len(unaugmented_ds), len(augmented_ds)])
                g.attrs['extractor'] = extractor_string
        elif model_name == "hoptimus0":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 1536, all_feats.shape
        elif model_name == "plip":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 512, all_feats.shape
        elif model_name == "biomedclip":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 512, all_feats.shape
        elif model_name == "dinosslpath":
            assert len(all_feats.shape) == 2 and all_feats.shape[1] == 384, all_feats.shape
        else:
            print(f"Model name did not match any known patterns: {model_name}")
            print(f"Shape of all_feats: {all_feats.shape}")
            raise ValueError(f"Unknown model name: {model_name}")
        f['feats'] = all_feats.cpu().numpy()
        f['augmented'] = np.repeat(
            [False, True], [len(unaugmented_ds), len(augmented_ds)])
        assert len(f['feats']) == len(f['augmented'])
        f.attrs['extractor'] = extractor_string

def feature_extract(output_dir: Path, wsi_dir: Path, model_path: Path, dataset: str, norm: bool,
               target_microns: int = 256, patch_size: int = 224, keep_dir_structure: bool = False,
               device: str = "cuda", normalization_template: Path = None, feat_extractor: str = "ctp"):
    # Clean up potentially old leftover .lock files

    has_gpu = torch.cuda.is_available()
    target_mpp = target_microns/patch_size

    # Initialize the feature extraction model
    print(f"Initialising feature extractor {feat_extractor}...")
    if feat_extractor == "ctp":
        extractor = FeatureExtractorCTP(checkpoint_path=model_path)
    elif feat_extractor == "chief-ctp":
        extractor = FeatureExtractorChiefCTP(checkpoint_path=model_path)
    elif feat_extractor == "uni":
        extractor = FeatureExtractorUNI()
    elif feat_extractor == "provgp":
        extractor = FeatureExtractorProvGP()
    elif feat_extractor == "hibou-b":
        extractor = FeatureExtractorHibouB()
    elif feat_extractor == "hibou-l":
        extractor = FeatureExtractorHibouL()
    elif feat_extractor == "kaiko":
        extractor = FeatureExtractorKaiko()
    elif feat_extractor == "conch":
        extractor = FeatureExtractorConch()
    elif feat_extractor == "phikon":
        extractor = FeatureExtractorPhikon()
    elif feat_extractor == "virchow":
        extractor = FeatureExtractorVirchow()
    elif feat_extractor == "virchow2":
        extractor = FeatureExtractorVirchow2()
    elif feat_extractor == "hoptimus0":
        extractor = FeatureExtractorHOptimus0()
    elif feat_extractor == "plip":
        extractor = FeatureExtractorPLIP()
    elif feat_extractor == "biomedclip":
        extractor = FeatureExtractorBiomedCLIP()
    elif feat_extractor == "dinosslpath":
        extractor = FeatureExtractorDinoSSLPath()
    else:
        raise Exception(f"Invalid feature extractor '{feat_extractor}' selected")

    model_name = extractor.init_feat_extractor(device=device)
    # Create cache and output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_method = "STAMP_macenko_" if norm else "STAMP_raw_"
    model_name_norm = Path(norm_method + model_name)
    output_file_dir = output_dir/model_name_norm/dataset
    output_file_dir.mkdir(parents=True, exist_ok=True)
    # Create logfile and set up logging
    logfile_name = "logfile_" + time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(os.getpid())
    logdir = output_file_dir/logfile_name
    logging.basicConfig(filename=logdir, force=True, level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Preprocessing started at: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(f"Norm: {norm} | Target_microns: {target_microns} | Patch_size: {patch_size} | MPP: {target_mpp}")
    logging.info(f"Model: {model_name}\n")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Stored logfile in {logdir}")
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    print(f"GPU is available: {has_gpu}")
    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}, using device {device}")

    if norm:
        print("\nInitialising Macenko normaliser...")
        print(normalization_template)
        target = cv2.imread(str(normalization_template))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        normalizer = utils.Normalizer()
        normalizer.fit(target)


    # Get list of slides, filter out slides that have already been processed
    slide_dir = wsi_dir/dataset/'output'
    
    img_dir = [item for item in slide_dir.iterdir() if item.is_dir()]
    for slide_url in tqdm(img_dir, "\nPreprocessing progress", leave=False, miniters=1, mininterval=0):
        slide_name = slide_url.stem
        start_time = time.time()
        feat_out_dir = output_file_dir/slide_name

        print("\n")
        logging.info(f"===== Processing slide {slide_name} =====")

        if not os.path.exists(f"{feat_out_dir}.h5") :
            extract_features_(model=extractor.model, transform=extractor.transform, model_name=model_name,
                                slide_url=slide_url,
                                outdir=feat_out_dir,  is_norm=norm, device=device if has_gpu else "cpu",
                                target_microns=target_microns, patch_size=patch_size,
                                processor = extractor.processor if (feat_extractor == "hibou-l" or feat_extractor == "hibou-b" or feat_extractor == "conch" or feat_extractor == "phikon" or feat_extractor == "plip" or feat_extractor == "biomedclip") else None)
    else:
        if os.path.exists((f"{feat_out_dir}.h5")):
            logging.info(".h5 file for this slide already exists. Skipping...")
        else:
            logging.info("Slide is already being processed. Skipping...")