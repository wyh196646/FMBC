from fastai.vision.all import Path, get_image_files, verify_images

from typing import Any, Optional, Callable, Tuple

from PIL import Image
from pathlib import Path
from dinov2.data.datasets.extended import ExtendedVisionDataset
import torch
from PIL import ImageFile, Image

import re
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
                 target_transform: Optional[Callable] = None,
                 dataset_list: list = [],
                 split: Optional[str] = 'train'
                 ) -> None:

        super().__init__(root, transforms, transform, target_transform)
        self.split = split
        self.root = Path(root).expanduser()
        self.dataset_list = dataset_list
        self.image_paths=[]
        
        if split == 'train':
            if len(self.dataset_list)==0:
                self.dataset_list = os.listdir(self.root) 
            print('processing',self.dataset_list)
            for subfolder in self.dataset_list:
                self.train_folder=os.path.join(self.root, subfolder ,'output')
                image_paths = get_image_files(self.train_folder)
                invalid_images = set()
                if verify_images:
                    print("Verifying images. This ran at ~100 images/sec/cpu for me. Probably depends heavily on disk perf.")
                    invalid_images = set(verify_images(image_paths))
                    print("Skipping invalid images:", invalid_images)
                self.image_paths.extend([p for p in image_paths if p not in invalid_images])
        else:
            self.image_paths = get_image_files(self.root)

    def get_image_data(self, index: int) -> bytes:  # should return an image as an array

        image_path = self.image_paths[index]
        coords = [int(num) for num in re.findall(r'\d+', image_path.name)]
        img = Image.open(image_path).convert(mode="RGB")

        return img, coords

    def get_target(self, index: int) -> Any:
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # try:
        image, coords = self.get_image_data(index)
        # except :
        #     image = Image.new(mode="RGB", size=(256, 256), color= (255, 255, 255))
        #     coords = [0,0]
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        #print(image.size)
        if self.split == 'train':
            return image, target
        else:
            return {
                'sample':[image,target],
                'coords':coords
            }

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)
    