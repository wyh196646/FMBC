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
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import h5py
import h5py
import numpy as np
import matplotlib.pyplot as plt


class SlideEmbeddingDataset(Dataset):
    def __init__(self,
                 embedding_root: str,
                 transform: callable = None,
                 num_transforms: int = 1,
                 ):
        
        self.feature_root = embedding_root
        self.feature_path= [f for f in Path(embedding_root).rglob('*.h5')]

        self.transform_ = transform
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
        if self.transform_:
            instance = self.transform_(assets)
        return assets, attrs
    

class SlideEmebddingMask(SlideEmbeddingDataset):
    def __init__(self, *args, embedding_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(SlideEmebddingMask, self).__init__(*args, **kwargs)
        self.psz = embedding_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

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

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        assets, attrs = self.read_assets_from_h5(self.feature_path[index]) 

        instance = self.transform_(assets)


        masks = []
        num_boxes = instance['global_box'].shape[0]
        high = int(self.get_pred_ratio() * num_boxes)
        mask = torch.cat([
            torch.zeros(num_boxes - high, dtype=torch.bool),
            torch.ones(high, dtype=torch.bool)
        ])
        mask = mask[torch.randperm(num_boxes)]  # 随机打乱mask

        masks.append(mask)
        
        # 更新instance字典
        instance.update({
            'masks': masks,
        })
        
        return instance

    
class BoundingBoxTransform:
    def __init__(self,
                 global_coverage_ratio = 0.4,
                 global_crops_number = 2,
                 local_coverage_ratio = 0.6 , 
                 local_crops_number = 8,
                 num_cluster = 8,
                 max_try_iter=200
                 ):
        
        self.global_coverage_ratio = global_coverage_ratio
        self.global_crops_number = global_crops_number
        self.local_coverage_ratio = local_coverage_ratio
        self.local_crops_number = local_crops_number
        self.num_cluster = num_cluster
        self.max_try_iter = max_try_iter
    

    
    @staticmethod
    def kmeans_clustering(features, n_clusters = 8):
        kmeans = KMeans(n_clusters = n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        return cluster_labels
    
    def generate_global_box(self, class_points, class_features):
        """
        Generate a global bounding box for a given class.

        Args:
            class_points (Tensor): Points belonging to a specific class (N x 2).
            class_features (Tensor): Features corresponding to class_points (N x F).

        Returns:
            Tuple:0
                - global_box_points (Tensor): Points inside the global box.
                - global_box_features (Tensor): Features inside the global box.
        """
        num_points = class_points.size(0)
        target_points = int(num_points * self.global_coverage_ratio)

        x_min, x_max = class_points[:, 0].min(), class_points[:, 0].max()
        y_min, y_max = class_points[:, 1].min(), class_points[:, 1].max()

        for _ in range(self.max_try_iter):
            x_min_rand = torch.FloatTensor(1).uniform_(x_min, x_max).item()
            y_min_rand = torch.FloatTensor(1).uniform_(y_min, y_max).item()
            width = torch.FloatTensor(1).uniform_(0, x_max - x_min).item()
            height = torch.FloatTensor(1).uniform_(0, y_max - y_min).item()

            x_max_rand = x_min_rand + width
            y_max_rand = y_min_rand + height

            box_points_mask = (
                (class_points[:, 0] >= x_min_rand) &
                (class_points[:, 0] <= x_max_rand) &
                (class_points[:, 1] >= y_min_rand) &
                (class_points[:, 1] <= y_max_rand)
            )
            box_points = class_points[box_points_mask]
            box_features = class_features[box_points_mask]

            if box_points.size(0) >= target_points:
                return box_points, box_features

        return torch.empty(0, 2), torch.empty(0, class_features.size(1))  # Fallback in case no box is found

    def generate_local_boxes(self, 
                             global_points,
                             global_features, 
                             x_min_rand, 
                             x_max_rand, 
                             y_min_rand, 
                             y_max_rand):
        """
        Generate local bounding boxes within the global bounding box.

        Args:
            global_points (Tensor): Points inside the global bounding box (N x 2).
            global_features (Tensor): Features inside the global bounding box (N x F).
            x_min_rand (float): Minimum x-coordinate of the global box.
            x_max_rand (float): Maximum x-coordinate of the global box.
            y_min_rand (float): Minimum y-coordinate of the global box.
            y_max_rand (float): Maximum y-coordinate of the global box.

        Returns:
            List[Tensor]: List of local boxes (each containing points).
            List[Tensor]: List of local features corresponding to the boxes.
        """
        local_boxes = []
        local_features = []
        target_points = int(global_points.size(0) * self.local_coverage_ratio)

        for _ in range(self.local_crops_number):
            is_empty = False
            for _ in range(self.max_try_iter):
                x_min_local = torch.FloatTensor(1).uniform_(x_min_rand, x_max_rand).item()
                y_min_local = torch.FloatTensor(1).uniform_(y_min_rand, y_max_rand).item()
                local_width = torch.FloatTensor(1).uniform_(0, x_max_rand - x_min_local).item()
                local_height = torch.FloatTensor(1).uniform_(0, y_max_rand - y_min_local).item()

                x_max_local = x_min_local + local_width
                y_max_local = y_min_local + local_height

                local_box_points_mask = (
                    (global_points[:, 0] >= x_min_local) &
                    (global_points[:, 0] <= x_max_local) &
                    (global_points[:, 1] >= y_min_local) &
                    (global_points[:, 1] <= y_max_local)
                )
                local_box_points = global_points[local_box_points_mask]
                local_box_features = global_features[local_box_points_mask]

                if local_box_points.size(0) >= target_points:
                    local_boxes.append(local_box_points)
                    local_features.append(local_box_features)
                    is_empty = True
                    break
            if not is_empty:
                local_boxes.append(torch.empty(0, 2))
                local_features.append(torch.empty(0, global_features.size(1)))
                
        return local_boxes, local_features

    def process_class(self, class_points, class_features):
        """
        Process a single class to generate global and local boxes.

        Args:
            class_points (Tensor): Points belonging to a specific class (N x 2).
            class_features (Tensor): Features corresponding to class_points (N x F).

        Returns:
            Tuple:
                - global_points (Tensor): Points inside the global box.
                - global_features (Tensor): Features inside the global box.
                - local_boxes (List[Tensor]): List of local boxes (each containing points).
                - local_features (List[Tensor]): List of local features corresponding to the boxes.
        """
        global_points, global_features = self.generate_global_box(class_points, class_features)

        if global_points.size(0) == 0:
            return global_points, global_features, [], []

        x_min, x_max = global_points[:, 0].min(), global_points[:, 0].max()
        y_min, y_max = global_points[:, 1].min(), global_points[:, 1].max()

        local_boxes, local_features = self.generate_local_boxes(
            global_points, global_features, x_min.item(), x_max.item(), y_min.item(), y_max.item()
        )

        return global_points, global_features, local_boxes, local_features

    def process_all_classes(self, points, features, labels):
        """

        Args:
            points (Tensor): Coordinates of all points (N x 2).
            features (Tensor): Features of all points (N x F).
            labels (Tensor): Cluster labels for the points (N).

        Returns:
            Tuple:
                - global_boxes (List[Tensor]): List of global boxes for all classes.
                - global_features (List[Tensor]): List of global features for all classes.
                - local_boxes (List[List[Tensor]]): List of local boxes for each class.
                - local_features (List[List[Tensor]]): List of local features for each class.
        """
        global_boxes = []
        local_boxes = []
        global_features = []
        local_features = []

        num_classes = len(torch.unique(labels))

        for class_id in range(num_classes):
            class_points = points[labels == class_id]
            class_features = features[labels == class_id]

            global_points, global_feature, local_box_points, local_box_features = self.process_class(
                class_points, class_features
            )

            if global_points.size(0) > 0:
                global_boxes.append(global_points)
                global_features.append(global_feature)
                local_boxes.append(local_box_points)
                local_features.append(local_box_features)
                
        global_boxes = torch.cat(global_boxes, dim=0)
        global_features = torch.cat(global_features, dim=0)
        local_boxes = self.clean_local_view(local_boxes)
        local_features = self.clean_local_view(local_features)
    
        return global_boxes, global_features, local_boxes, local_features

    @staticmethod
    def clean_local_view(data):
        result = []

        for i in range(len(data[0])):
            valid_tensors = [data[j][i] for j in range(len(data)) 
                            if data[j][i].numel() != 0] 
            if valid_tensors:  
                result.append(torch.cat(valid_tensors, dim=0))
        
        return result
        
    def __call__(self, data):
        points = data['coords']
        features = data['features']
        cluster_labels = self.kmeans_clustering(features, self.num_cluster)

        labels = torch.tensor(cluster_labels)
        points = torch.tensor(points)
        features = torch.tensor(features)
        global_boxes, global_features, local_boxes, local_features = self.process_all_classes(
            points, features, labels
        )
        return {
            'global_box': global_boxes,
            'global_feature': global_features,
            'local_box': local_boxes,
            'local_feature': local_features,
        }


def pad_tensors(imgs, coords):
    max_len = max([t.size(0) for t in imgs])  # get the maximum length
    padded_tensors = []
    padded_coords = []
    masks = []
    for i in range(len(imgs)):
        tensor = imgs[i]
        coord = coords[i]
        N_i = tensor.size(0)
        padded_tensor = torch.zeros(max_len, tensor.size(1))
        padded_coord = torch.zeros(max_len, 2)
        mask = torch.zeros(max_len)
        padded_tensor[:N_i] = tensor
        padded_coord[:N_i] = coord
        mask[:N_i] = 1
        padded_tensors.append(padded_tensor)
        padded_coords.append(padded_coord)
        masks.append(mask)
    return torch.stack(padded_tensors), torch.stack(padded_coords), torch.stack(masks).bool()

def pad_local_tensors(local_list):
    """
    Pad a list of local tensors to the same dimension within a sample.
    :param local_list: list of tensors, each tensor shape: [L, D]
    :return: padded_tensor: [num_views, max_L, D], mask: [num_views, max_L]
    """
    max_views = len(local_list)
    max_len = max([t.size(0) for t in local_list])
    feature_dim = local_list[0].size(1)

    padded_local = torch.zeros(max_views, max_len, feature_dim)
    masks = torch.zeros(max_views, max_len)

    for i, local_tensor in enumerate(local_list):
        L = local_tensor.size(0)
        padded_local[i, :L] = local_tensor
        masks[i, :L] = 1

    return padded_local, masks.bool()

def pad_predict_mask(tensor_list, pad_value=0):

    max_length = max(tensor.size(0) for tensor in tensor_list)

    # 创建一个填充后的tensor列表
    padded_tensors = []
    for t in tensor_list:
        pad_length = max_length - t.size(0)  # 计算需要填充的长度
        padded_t = torch.nn.functional.pad(t, (0, pad_length), "constant", pad_value)  # 使用指定的值填充
        padded_tensors.append(padded_t)

    # 堆叠成一个张量
    return torch.stack(padded_tensors)

def box_collate_fn(batch):
    """
    Custom collate_fn to pad global and local features/boxes.
    :param batch: list of samples, each sample is a dict:
                  {
                    'global_box': Tensor[L_g, 2],
                    'global_feature': Tensor[L_g, D_g],
                    'local_box': [Tensor[L_l1, 2], Tensor[L_l2, 2], ...],
                    'local_feature': [Tensor[L_l1, D_l], Tensor[L_l2, D_l], ...]
                  }
    :return: 
        - padded_global_boxes: Tensor[B, max_Lg, 2]
        - padded_global_features: Tensor[B, max_Lg, D_g]
        - global_masks: Tensor[B, max_Lg]
        - padded_local_boxes: Tensor[B, num_views, max_Ll, 2]
        - padded_local_features: Tensor[B, num_views, max_Ll, D_l]
        - local_masks: Tensor[B, num_views, max_Ll]
    """
    global_boxes_list = []
    global_features_list = []
    local_boxes_list = []
    local_features_list = []
    predict_mask = []
   
    for sample in batch:
        # Global padding
        global_boxes_list.append(sample['global_box'])
        global_features_list.append(sample['global_feature'])
        predict_mask.append(sample['predict_mask'])

        # Local padding within a sample
        local_boxes_padded, _ = pad_local_tensors(sample['local_box'])
        local_features_padded, local_masks = pad_local_tensors(sample['local_feature'])

        local_boxes_list.append(local_boxes_padded)
        local_features_list.append(local_features_padded)

    #padding the predict mask
    predict_massk= pad_predict_mask(predict_mask)
    # Global padding across the batch
    padded_global_features, padded_global_boxes, global_masks = pad_tensors(global_features_list, global_boxes_list)
    
    # Local padding across the batch
    max_num_views = max([t.size(0) for t in local_boxes_list])
    max_len_local = max([t.size(1) for t in local_boxes_list])
    local_feature_dim = local_features_list[0].size(2)
    local_box_dim = local_boxes_list[0].size(2)

    padded_local_boxes = torch.zeros(len(batch), max_num_views, max_len_local, local_box_dim)
    padded_local_features = torch.zeros(len(batch), max_num_views, max_len_local, local_feature_dim)
    local_masks = torch.zeros(len(batch), max_num_views, max_len_local)

    for i in range(len(batch)):
        num_views = local_boxes_list[i].size(0)
        len_local = local_boxes_list[i].size(1)
        padded_local_boxes[i, :num_views, :len_local] = local_boxes_list[i]
        padded_local_features[i, :num_views, :len_local] = local_features_list[i]
        local_masks[i, :num_views, :len_local] = 1

    return {
        'padded_global_boxes': padded_global_boxes,
        'padded_global_features': padded_global_features,
        'global_masks': global_masks,
        'padded_local_boxes': padded_local_boxes,
        'padded_local_features': padded_local_features,
        'local_masks': local_masks.bool(),
        'predict_mask': predict_massk
    }
    
    
    

if __name__ == '__main__':
    test_case='/home/yuhaowang/data/embedding/TCGA-BRCA/TCGA-AR-A1AL-01Z-00-DX1.h5'
    temp = read_assets_from_h5(test_case)[0]
    transform = BoundingBoxTransform()
    transformed = transform(temp)