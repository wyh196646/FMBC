from fastai.vision.all import Path, get_image_files, verify_images

from typing import Any, Optional, Callable, Tuple

from PIL import Image
from pathlib import Path
from dinov2.data.datasets.extended import ExtendedVisionDataset
import torch
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os 
def default_image(size=(256, 256)):
    return torch.zeros(3, *size)

class TileDataset(ExtendedVisionDataset):
    def __init__(self,
                 root: str,
                 verify_images: bool = False,
                 transforms: Optional[Callable] = None,
                    transform: Optional[Callable] = None,
                    target_transform: Optional[Callable] = None,split: str = 'train',) -> None:

        super().__init__(root, transforms, transform, target_transform)

        self.root = Path(root).expanduser()
        self.image_paths=[]
        # subfolder_list=['private_chunk_1',
        #                 'private_chunk_2',
        #                 'private_chunk_3',
        #                 'private_chunk_4',
        #                 'private_chunk_5',
        #                 'private_chunk_6',
        #                 'private_chunk_7',
        #                 'private_chunk_8',
        #                 'private_chunk_9',
        #                 'private_chunk_10',
        #                 ]
        subfolder_list=os.listdir(self.root)
        #subfolder_list=['TCGA-BRCA']
        print('processing',subfolder_list)
        for subfolder in subfolder_list:
            self.train_folder=os.path.join(self.root, subfolder ,'output')
            image_paths = get_image_files(self.train_folder)
            invalid_images = set()
            if verify_images:
                print("Verifying images. This ran at ~100 images/sec/cpu for me. Probably depends heavily on disk perf.")
                invalid_images = set(verify_images(image_paths))
                print("Skipping invalid images:", invalid_images)
            self.image_paths.extend([p for p in image_paths if p not in invalid_images])
        # self.train_folder =os.path.join(self.root, 'output')
        # image_paths = get_image_files(self.train_folder)
        # invalid_images = set()
        # if verify_images:
        #     print("Verifying images. This ran at ~100 images/sec/cpu for me. Probably depends heavily on disk perf.")
        #     invalid_images = set(verify_images(image_paths))
        #     print("Skipping invalid images:", invalid_images)
        # self.image_paths.extend([p for p in image_paths if p not in invalid_images])
        

    def get_image_data(self, index: int) -> bytes:  # should return an image as an array

        image_path = self.image_paths[index]
        img = Image.open(image_path).convert(mode="RGB")

        return img

    def get_target(self, index: int) -> Any:
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except :
            image= Image.new(mode="RGB", size=(256, 256), color= (255, 255, 255))
            #raise IndexError(f"can not read image for sample {index}")
            #raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)
    
   #source activate dinov2 && cd /ruiyan/yuhao/project/dinov2 && CUDA_VISIBLE_DEVICES=0,1,2, python -m torch.distributed.launch --master_port 11120  --nproc_per_node=1 dinov2/train/train.py --config-file=dinov2/configs/train/patch.yaml --output-dir=./output/ train.dataset_path=TileDataset:split=TRAIN:root=/ruiyan/yuhao/data