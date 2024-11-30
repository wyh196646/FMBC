import torch

from torch import nn
from . import slide_encoder
import sys
from pathlib import Path
import os
import math
import numpy as np
import warnings
sys.path.append(os.path.join(str(Path(__file__).resolve().parents[2]),'dino_stage2'))
import vision_transformer as vits

def reshape_input(imgs, coords, pad_mask=None):
    if len(imgs.shape) == 4:
        imgs = imgs.squeeze(0)
    if len(coords.shape) == 4:
        coords = coords.squeeze(0)
    if pad_mask is not None:
        if len(pad_mask.shape) != 2:
            pad_mask = pad_mask.squeeze(0)
    return imgs, coords, pad_mask

def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        model.eval()
        return model
        
class ClassificationHead(nn.Module):
    """
    The classification head for the slide encoder

    Arguments:
    ----------
    input_dim: int
        The input dimension of the slide encoder
    latent_dim: int
        The latent dimension of the slide encoder
    feat_layer: str
        The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers
    n_classes: int
        The number of classes
    model_arch: str
        The architecture of the slide encoder
    pretrained: str
        The path to the pretrained slide encoder
    freeze: bool
        Whether to freeze the pretrained model
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        feat_layer,
        n_classes=2,
        model_arch="vit_small",
        pretrained="",
        freeze=False,
        **kwargs,
    ):
        super(ClassificationHead, self).__init__()

        # setup the slide encoder
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        #self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)
        self.slide_encoder=vits.__dict__[model_arch](embed_dim=input_dim)
        load_pretrained_weights(self.slide_encoder, pretrained, 'teacher')
        # whether to freeze the pretrained model
        if freeze:
            print("Freezing Pretrained GigaPath model")
            for name, param in self.slide_encoder.named_parameters():
                param.requires_grad = False
            print("Done")
        # setup the classifier
        #num layers of slide_encoder
        self.encoder_num_layers= len(list(self.slide_encoder.named_parameters()))
        self.classifier = nn.Sequential(*[nn.Linear(latent_dim, n_classes)])
        #
    def forward(self, images: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
        ----------
        images: torch.Tensor
            The input images with shape [N, L, D]
        coords: torch.Tensor
            The input coordinates with shape [N, L, 2]
        """
        # inputs: [N, L, D]
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        assert len(images.shape) == 3
        mask=torch.zeros(images.shape[0], images.shape[1]+1, dtype=torch.bool).to(device=images.device)
        img_enc = self.slide_encoder((images, mask))
        logits = self.classifier(img_enc)
        return logits


def get_model(**kwargs):
    model = ClassificationHead(**kwargs)
    return model



