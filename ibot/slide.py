import random
import math
import numpy as np
import os
import torch
import torch.utils.data as data
import h5py
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.cluster import KMeans

class ClusterTransform:
    def __init__(self,
                 global_coverage_ratio=0.4,
                 local_coverage_ratio=0.6,
                 num_cluster=8,
                 local_crops_number=8,
                 global_crops_number=2):
        self.global_coverage_ratio = global_coverage_ratio
        self.local_coverage_ratio = local_coverage_ratio
        self.num_cluster = num_cluster
        self.local_crops_number = local_crops_number
        self.global_crops_number = global_crops_number

    @staticmethod
    def kmeans_clustering(features, n_clusters=8):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        return cluster_labels

    @staticmethod
    def find_nearest_neighbors(points, seed_idx, target_count):
        """ Finds the nearest neighbors to a seed point. """
        distances = torch.sum((points - points[seed_idx]) ** 2, dim=1)
        sorted_indices = torch.argsort(distances)
        return sorted_indices[:target_count]

    def generate_coords(self, points, features, coverage_ratio):
        """Generate coords by selecting nearest neighbors."""
        num_points = points.size(0)
        target_points = int(num_points * coverage_ratio)

        seed_idx = torch.randint(0, num_points, (1,)).item()
        nearest_indices = self.find_nearest_neighbors(points, seed_idx, target_points)
        return points[nearest_indices], features[nearest_indices]

    def __call__(self, data):
        points = torch.tensor(data['coords'])
        features = torch.tensor(data['features'])
        labels = torch.tensor(self.kmeans_clustering(features.numpy(), self.num_cluster))

        global_coords_list = [[] for _ in range(self.global_crops_number)]
        global_features_list = [[] for _ in range(self.global_crops_number)]
        local_coords_list = [[] for _ in range(self.local_crops_number)]
        local_features_list = [[] for _ in range(self.local_crops_number)]

        for crop_idx in range(self.global_crops_number):
            for cluster_id in torch.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_points = points[cluster_mask]
                cluster_features = features[cluster_mask]

                if cluster_points.size(0) > 0:
                    global_coords, global_features = self.generate_coords(
                        cluster_points, cluster_features, self.global_coverage_ratio
                    )
                    global_coords_list[crop_idx].append(global_coords)
                    global_features_list[crop_idx].append(global_features)

        for crop_idx in range(self.local_crops_number):
            for cluster_id in torch.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_points = points[cluster_mask]
                cluster_features = features[cluster_mask]

                if cluster_points.size(0) > 0:
                    local_coords, local_features = self.generate_coords(
                        cluster_points, cluster_features, self.local_coverage_ratio
                    )
                    local_coords_list[crop_idx].append(local_coords)
                    local_features_list[crop_idx].append(local_features)

        return {
            'global_coords': [torch.cat(crop) for crop in global_coords_list],
            'global_features': [torch.cat(crop) for crop in global_features_list],
            'local_coords': [torch.cat(crop) for crop in local_coords_list],
            'local_features': [torch.cat(crop) for crop in local_features_list]
        }
        
class SlideEmbeddingDataset(Dataset):
    def __init__(self,
                 embedding_root: str,
                 transform: callable = None,
                 num_transforms: int = 1,
                 ):
        
        self.feature_root = embedding_root
        self.feature_path= [f for f in Path(embedding_root).rglob('*.h5')]

        self.transform = transform
        self.num_transforms = num_transforms
        
        
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
        if self.transform:
            instance = self.transform(assets)
        return assets, attrs
    
class SlideEmbeddingMask(SlideEmbeddingDataset):
    def __init__(self, *args, patch_sizeembedding_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(SlideEmbeddingMask, self).__init__(*args, **kwargs)
        self.psz = patch_sizeembedding_size
        self.pred_ratio = pred_ratio  
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def generate_mask(self, num_points):
        high = int(self.get_pred_ratio() * num_points)
        mask = torch.cat([
            torch.zeros(num_points - high, dtype=torch.bool),
            torch.ones(high, dtype=torch.bool)
        ])
        mask = mask[torch.randperm(num_points)]
        return mask
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio
    
    def __getitem__(self, idx):
        data,_ = self.read_assets_from_h5(self.feature_path[idx])
        if self.transform:
            data = self.transform(data)
        predict_global_mask = []
        for global_coord in data['global_coords']:
            predict_global_mask.append(self.generate_mask(global_coord.size(0)))

        data['predict_global_mask'] = predict_global_mask

        return data

def pad_variable_length(tensors, max_len):
    padded_tensors = []
    masks = []
    for tensor in tensors:
        N_i = tensor.size(0)
        padded_tensor = torch.zeros(max_len, *tensor.size()[1:])
        mask = torch.zeros(max_len)

        padded_tensor[:N_i] = tensor
        mask[:N_i] = 1

        padded_tensors.append(padded_tensor)
        masks.append(mask)

    return torch.stack(padded_tensors), torch.stack(masks).bool()

def split_list(input_list, k):
    return [input_list[i:i + k] for i in range(0, len(input_list), k)]

def pad_coords_and_features(coords, features):
    max_len = max([coord.size(0) for coord in coords])
    padded_coords, coord_masks = pad_variable_length(coords, max_len)
    padded_features, feature_masks = pad_variable_length(features, max_len)
    return padded_coords, padded_features, coord_masks

def custom_collate_fn(batch):
    global_coords, global_features, predict_global_masks = [], [], []
    local_coords, local_features = [], []

    for sample in batch:
        global_coords.extend(sample['global_coords'])
        global_features.extend(sample['global_features'])
        predict_global_masks.extend(sample['predict_global_mask'])
        local_coords.extend(sample['local_coords'])
        local_features.extend(sample['local_features'])

    padded_global_coords, padded_global_features, global_attention_masks = pad_coords_and_features(global_coords, global_features)
    padded_predict_global_masks, _ = pad_variable_length(predict_global_masks, max(global_attention_masks.size(1), len(predict_global_masks)))
    padded_local_coords, padded_local_features, local_attention_masks = pad_coords_and_features(local_coords, local_features)
    # the global_attention_masks
    global_attention_masks = ~global_attention_masks
    local_attention_masks = ~local_attention_masks
    len_global = len(batch[0]['global_coords'])
    len_local = len(batch[0]['local_coords'])
    return {
        'padded_global_coords': split_list(padded_global_coords, len_global),
        'padded_global_features': split_list(padded_global_features, len_global),
        'global_attention_masks': split_list(global_attention_masks, len_global),
        'padded_predict_global_masks': padded_predict_global_masks,
        'padded_local_coords': split_list(padded_local_coords, len_local),
        'padded_local_features': split_list(padded_local_features, len_local),
        'local_attention_masks': split_list(local_attention_masks, len_local),
    }
    

if __name__ == '__main__':
    embedding_root = '/home/yuhaowang/data/embedding/TCGA-BRCA'
    transform = ClusterTransform(global_coverage_ratio=0.6, local_coverage_ratio=0.4, local_crops_number=4, global_crops_number=2)
    dataset = SlideEmbeddingMask(embedding_root, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)

    for batch in dataloader:
        print(batch['padded_global_coords'].shape)
        print(batch['global_attention_masks'].shape)
        print(batch['padded_predict_global_masks'].shape)
        print(batch['padded_local_coords'].shape)
        print(batch['local_attention_masks'].shape)
        break
