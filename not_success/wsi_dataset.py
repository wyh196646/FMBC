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
    def __init__(self, feature_root,teacher_ratio,student_ratio,
                 num_clusters,mlm_ratio,stage='train'):
        self.num_cluster = num_clusters
        self.feature_root = feature_root
        self.feature_path= [f for f in Path(feature_root).rglob('*.h5')]
        self.teacher_ratio = teacher_ratio
        self.student_ratio = student_ratio
        self.mlm_ratio=  mlm_ratio
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

        if self.stage=='train':
            teacher_feature, teacher_coords, teacher_mlm_mask = self.feature_select(
                assets, self.teacher_ratio, self.mlm_ratio)
            student_feature, student_coords, student_mlm_mask  = self.feature_select(
                assets, self.student_ratio, self.mlm_ratio)
            return torch.FloatTensor(student_feature),\
                    torch.FloatTensor(teacher_feature), \
                    torch.FloatTensor(student_coords),\
                    torch.FloatTensor(teacher_coords)
        else:
            return torch.FloatTensor(assets['feature'])
    
    def feature_select(self, assets, selct_ratio):
        clustering_dict= element_indices(assets['labels'])
        feature_index=[np.random.choice(value, int(len(value)*selct_ratio), replace=False) 
                    for _, value in clustering_dict.items()]
        feature_index=np.concatenate(feature_index)
        feature=assets['features'][feature_index]

        feature_coods=assets['coords'][feature_index]
        return feature, feature_coods
    
    
def collate_fn(batch):
    student_feature_list, teacher_feature_list,  student_coords_list, teacher_coords_list= zip(*batch)
    teacher_feature,teacher_coords, teachcer_mask = collate_branch(
        teacher_feature_list,teacher_coords_list)
    
    student_feature, student_coords, student_mask = collate_branch(
        student_feature_list, student_coords_list)
    return teacher_feature, teacher_coords, teachcer_mask, student_feature, \
            student_coords, student_mask

      

def collate_branch(features, feature_coords):
    batch_size = len(features)
    embedding_dim = features[0].shape[1]
    max_feature_len = max(element.shape[0] for element in features)
    feature_batch = torch.zeros(batch_size, max_feature_len , embedding_dim)
    for i, feature in enumerate(features):
        feature_batch[i, :feature.shape[0]] =feature
    # Generate length attention masks
    feature_lengths = torch.tensor([feature.shape[0] for feature in features])
    feature_mask = torch.arange(max_feature_len).expand(batch_size, -1) >= feature_lengths.unsqueeze(1)
    #Generate mlm mask

    # Add CLS token mask
    cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool)
    feacher_mask = torch.cat((cls_mask, feature_mask), dim=1)
    
    
    
    #padding max length of coordinates
    max_coords_len = max(element.shape[0] for element in feature_coords)
    padded_coords = torch.zeros((batch_size, max_coords_len, 2))
    for i, coords in enumerate(feature_coords):
        padded_coords[i, :coords.shape[0]] = coords
        
    return feature_batch, padded_coords, feacher_mask
    



    
