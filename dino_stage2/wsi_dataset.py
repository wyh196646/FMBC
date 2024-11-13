#define a dataset for whole slide image
import os
import numpy as np
import torch
import torch.utils.data as data
import openslide
import cv2
import random
import json
import math
from torch.utils.data import Dataset
from PIL import Image
import sys
from utils import element_indices
import h5py
import glob
from pathlib import Path
class WSIDataset(Dataset):
    def __init__(self, feature_root,num_clusters=50,represent_ratio=0.1,stage='train'):
        self.num_cluster = num_clusters
        self.feature_root = feature_root
        self.feature_path= [f for f in Path(feature_root).rglob('*.h5')]
        self.represent_ratio=represent_ratio
        self.stage=stage
        
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
        assets, _ = self.read_assets_from_h5(self.feature_path[idx])
        #print(assets)
        if self.stage=='train':
            clustering_dict= element_indices(assets['labels'])
            student_index=[np.random.choice(value, int(len(value)*self.represent_ratio), replace=False) 
                        for key, value in clustering_dict.items()]
            student_index=np.concatenate(student_index)
            student_feature=assets['features'][student_index]
            return torch.FloatTensor(student_feature),torch.FloatTensor(assets['cluster_centers'])
        else:
            return torch.FloatTensor(assets['feature'])
    
    
        



    



    
