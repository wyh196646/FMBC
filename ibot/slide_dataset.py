import random
import math
import numpy as np

import os
import numpy as np
import torch
import torch.utils.data as data
import openslide
import cv2
import random
import json
import math
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict, deque
from torchvision.datasets import ImageFolder
import h5py


class SlideEmbeddingDataset(Dataset):
    def __init__(self,
                 embedding_root: str,
                 teacher_ratio:int = 0.2 ,
                 student_ratio:float = 0.1,
                 num_clusters: int = 50,
                 transform: callable = None,
                 num_transforms: int = 1,
                 stage: str = 'train'):
        
        self.num_cluster = num_clusters
        self.feature_root = embedding_root
        self.feature_path= [f for f in Path(embedding_root).rglob('*.h5')]
        self.teacher_ratio = teacher_ratio
        self.student_ratio = student_ratio
        self.stage = stage
        self.num_transforms_ = num_transforms
        
    def __len__(self):
        return len(self.feature_path)
    
    def read_assets_from_h5(self, h5_path: str) -> tuple:
        '''Read the assets from the h5 file'''
        assets = {}
        attrs = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs
    
    def __getitem__(self, idx):
        assets, attrs = self.read_assets_from_h5(self.feature_path[idx])
        
        
        for transform_idx in range(self.num_transforms_):
            pt_path = random.choice(tag_list)
            inst_ = torch.load(pt_path)

            instance["embeddings"][transform_idx] = inst_["embeddings"]
            instance["coords"][transform_idx] = torch.tensor(inst_["coords"])

            del inst_

        if self.transform_:
            instance = self.transform_(instance)

        return instance
        return assets, attrs
        
    
    # @staticmethod
    # def element_indices(lst):
    #     index_dict = defaultdict(list)  
    #     for i, value in enumerate(lst):
    #         index_dict[value].append(i)  
    #     return dict(index_dict)
    
    # def __getitem__(self, idx):
    #     assets, _ = self.read_assets_from_h5(self.feature_path[idx])
    #     #print(assets)
    #     if self.stage=='train':
    #         teacher_feature, _ = self.feature_select(assets, self.teacher_ratio)
    #         student_feature, _ = self.feature_select(assets, self.student_ratio)
    #         return torch.FloatTensor(student_feature), torch.FloatTensor(teacher_feature)
    #     else:
    #         return torch.FloatTensor(assets['feature'])
    
    # def feature_select(self,assets,ratio):
    #     clustering_dict = self.element_indices(assets['labels'])
    #     student_index = [np.random.choice(value, int(len(value)*ratio)) 
    #                 for key, value in clustering_dict.items()]
    #     student_index = np.concatenate(student_index)
    #     student_feature = assets['features'][student_index]
    #     student_index=[np.random.choice(value, int(len(value)*ratio)) 
    #                 for key, value in clustering_dict.items()]
    #     return student_feature, student_index
    