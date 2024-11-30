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
    def __init__(self, feature_root,teacher_ratio=0.2,student_ratio=0.1,num_clusters=50,stage='train'):
        self.num_cluster = num_clusters
        self.feature_root = feature_root
        self.feature_path= [f for f in Path(feature_root).rglob('*.h5')]
        self.teacher_ratio = teacher_ratio
        self.student_ratio = student_ratio
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
            teacher_feature,_ = self.feature_select(assets,self.teacher_ratio)
            student_feature,_ = self.feature_select(assets,self.student_ratio)
            return torch.FloatTensor(student_feature),torch.FloatTensor(teacher_feature)
        else:
            return torch.FloatTensor(assets['feature'])
    
    def feature_select(self,assets,ratio):
        clustering_dict= element_indices(assets['labels'])
        student_index=[np.random.choice(value, int(len(value)*ratio), replace=False) 
                    for key, value in clustering_dict.items()]
        student_index=np.concatenate(student_index)
        student_feature=assets['features'][student_index]
        student_index=[np.random.choice(value, int(len(value)*ratio), replace=False) 
                    for key, value in clustering_dict.items()]
        return student_feature,student_index
    
    
        



    



    
