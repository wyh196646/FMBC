import os
import h5py
import torch
import pandas as pd
import numpy as np
import sys
from torch.utils.data import Dataset
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from typing import Any, Iterable, Optional, Sequence, Tuple, Union, Protocol, Callable
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','dino_stage2'))

class SlideDatasetForTasks(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 splits: list,
                 task_config: dict, 
                 slide_key: str='slide_id',
                 split_key: str='pat_id',
                 sampling_ratio: float=0.2,
                 **kwargs
                 ):
        '''
        This class is used to set up the slide dataset for different tasks

        Arguments:
        ----------
        data_df: pd.DataFrame
            The dataframe that contains the slide data
        root_path: str
            The root path of the tile embeddings
        splits: list
            The list of splits to use
        task_config: dict
            The task configuration dictionary
        slide_key: str
            The key that contains the slide id
        split_key: str
            The key that specifies the column for splitting the data
        '''
        self.root_path = root_path
        self.split_key = split_key
        self.slide_key = slide_key
        self.task_cfg = task_config
        self.sampling_ratio =sampling_ratio

        # get slides that have tile encodings
        valid_slides = self.get_valid_slides(root_path, data_df[slide_key].values)
        # filter out slides that do not have tile encodings
        data_df = data_df[data_df[slide_key].isin(valid_slides)]
        #print(data) 
        # set up the task
        self.setup_data(data_df, splits, task_config.get('setting', 'multi_class'))
        
        self.max_tiles = task_config.get('max_tiles', 1000)
        self.shuffle_tiles = task_config.get('shuffle_tiles', False)
        #print(self.labels)
        print('Dataset has been initialized!')
        
    def get_valid_slides(self, root_path: str, slides: list) -> list:
        '''This function is used to get the slides that have tile encodings stored in the tile directory'''
        valid_slides = []
        for i in range(len(slides)):
            if 'pt_files' in root_path.split('/')[-1]:
                sld = str(slides[i]).replace(".svs", "") + '.pt'
            else:
                sld = str(slides[i]).replace(".svs", "") + '.h5'
            sld_path = os.path.join(root_path, sld)
            if not os.path.exists(sld_path):
                print('Missing: ', sld_path)
            else:
                valid_slides.append(str(slides[i]))
                #print('valid: {} success'.format(sld_path))
        return valid_slides
    
    def setup_data(self, df: pd.DataFrame, splits: list, task: str='multi_class'):
        '''Prepare the data for multi-class setting or multi-label setting'''
        # Prepare slide data
        if task == 'multi_class' or task == 'binary':
            prepare_data_func = self.prepare_multi_class_or_binary_data
        elif task == 'multi_label':
            prepare_data_func = self.prepare_multi_label_data
        elif task == 'regression':
            prepare_data_func = self.prepare_regression_data
        elif task == 'survival':
            prepare_data_func = self.prepare_survival_data
        else:
            raise ValueError('Invalid task: {}'.format(task))
        self.slide_data, self.images, self.labels, self.n_classes = prepare_data_func(df, splits)
    
    def prepare_regression_data(self, df: pd.DataFrame, splits: list):
        '''Prepare the data for regression'''
        label_dict = self.task_cfg.get('label_dict', {})
        assert label_dict, 'No label_dict found in the task configuration'
        # Prepare mutation data
        label_keys = label_dict.keys()
        # sort key using values
        label_keys = sorted(label_keys, key=lambda x: label_dict[x])
        n_classes = len(label_dict)

        # get the corresponding splits
        assert self.split_key in df.columns, 'No {} column found in the dataframe'.format(self.split_key)
        df = df[df[self.split_key].isin(splits)]

        #TCGA-BH-A1EO-01Z-00-DX1
        images = df[self.slide_key].to_list()
        labels = df[label_keys].to_numpy()
        
        return df, images, labels, n_classes
    def prepare_survival_data(self, df: pd.DataFrame, splits: list):
        '''Prepare the data for regression'''
        label_dict = self.task_cfg.get('label_dict', {})
        assert label_dict, 'No label_dict found in the task configuration'
        # Prepare mutation data
        label_keys = label_dict.keys()
        # sort key using values
        label_keys = sorted(label_keys, key=lambda x: label_dict[x])

        # get the corresponding splits
        assert self.split_key in df.columns, 'No {} column found in the dataframe'.format(self.split_key)
        df = df[df[self.split_key].isin(splits)]
        images = df[self.slide_key].to_list()
        labels = df[label_keys].to_numpy().astype(int)
        n_bins = self.task_cfg.get('n_bins', 4)
            
        return df, images, labels, n_bins
    
        
    def prepare_multi_class_or_binary_data(self, df: pd.DataFrame, splits: list):
        '''Prepare the data for multi-class classification'''
        # set up the label_dict
        label_dict = self.task_cfg.get('label_dict', {})
        #print(label_dict)
        #print(df)
        assert  label_dict, 'No label_dict found in the task configuration'
        # set up the mappings
        assert 'label' in df.columns, 'No label column found in the dataframe'
        df['label'] = df['label'].map(label_dict)
        #print(df['label'])
        n_classes = len(label_dict)
        
        # get the corresponding splits
        assert self.split_key in df.columns, 'No {} column found in the dataframe'.format(self.split_key)
        df = df[df[self.split_key].isin(splits)]
       #print(df)
        images = df[self.slide_key].to_list()
        #print(df[['label']])
        labels = df[['label']].to_numpy().astype(int)
        #print(labels)
        #print(df[['label']].to_numpy())
        return df, images, labels, n_classes
        
    def prepare_multi_label_data(self, df: pd.DataFrame, splits: list):
        '''Prepare the data for multi-label classification'''
        # set up the label_dict
        label_dict = self.task_cfg.get('label_dict', {})
        assert label_dict, 'No label_dict found in the task configuration'
        # Prepare mutation data
        label_keys = label_dict.keys()
        # sort key using values
        label_keys = sorted(label_keys, key=lambda x: label_dict[x])
        n_classes = len(label_dict)

        # get the corresponding splits
        assert self.split_key in df.columns, 'No {} column found in the dataframe'.format(self.split_key)
        df = df[df[self.split_key].isin(splits)]
        images = df[self.slide_key].to_list()
        labels = df[label_keys].to_numpy().astype(int)
            
        return df, images, labels, n_classes
    
    
class SlideDataset(SlideDatasetForTasks):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 splits: list,
                 task_config: dict,
                 slide_key='slide_id',
                 split_key='pat_id',
                 **kwargs
                 ):
        '''
        The slide dataset class for retrieving the slide data for different tasks

        Arguments:
        ----------
        data_df: pd.DataFrame
            The dataframe that contains the slide data
        root_path: str
            The root path of the tile embeddings
        splits: list
            The list of splits to use
        task_config_path: dict
            The task configuration dictionary
        slide_key: str
            The key that contains the slide id
        split_key: str
            The key that specifies the column for splitting the data
        '''
        super(SlideDataset, self).__init__(data_df, root_path, splits, task_config, slide_key, split_key, **kwargs)

    def shuffle_data(self, images: torch.Tensor, coords: torch.Tensor) -> tuple:
        '''Shuffle the serialized images and coordinates'''
        indices = torch.randperm(len(images))
        images_ = images[indices]
        coords_ = coords[indices]
        return images_, coords_

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
    
    def get_sld_name_from_path(self, sld: str) -> str:
        '''Get the slide name from the slide path'''
        sld_name = os.path.basename(sld).split('.h5')[0]
        return sld_name
    
    def get_images_from_path(self, img_path: str) -> dict:
        '''Get the images from the path'''
        if '.pt' in img_path:
            images = torch.load(img_path)
            coords = 0
        elif '.h5' in img_path:
            assets, _ = self.read_assets_from_h5(img_path)
            images = torch.from_numpy(assets['features'])
            #print(images.shape)
            try:
                coords = torch.from_numpy(assets['coords'])
            except :
                coords = torch.zeros((1, 2))
            #print(coords.shape)  
            if self.shuffle_tiles and coords.shape[0]>1:
                images, coords = self.shuffle_data(images, coords)
            if images.size(0) > self.max_tiles:
                images = images[:self.max_tiles, :]
            if coords.size(0) > self.max_tiles:
                coords = coords[:self.max_tiles, :]
        
        # set the input dict
        data_dict = {'imgs': images,
                'img_lens': images.size(0),
                'coords': coords}
        #print(data_dict['imgs'].shape)
        return data_dict
    
    def get_balance_weight(self):
        # for data balance
        label = self.labels
        label_np = np.array(label)
        classes = list(set(label))
        N = len(self.df)
        num_of_classes = [(label_np==c).sum() for c in classes]
        c_weight = [N/num_of_classes[i] for i in range(len(classes))]
        weight = [0 for _ in range(N)]
        for i in range(N):
            c_index = classes.index(label[i])
            weight[i] = c_weight[c_index]
        return weight
    
    def get_one_sample(self, idx: int) -> dict:
        '''Get one sample from the dataset'''
        # get the slide id

        slide_id = self.images[idx]
        # get the slide path
        if 'pt_files' in self.root_path.split('/')[-1]:
            slide_path = os.path.join(self.root_path, slide_id.replace(".svs", "") + '.pt')
        else:
            slide_path = os.path.join(self.root_path, slide_id.replace(".svs", "") + '.h5')
        # get the slide images
        data_dict = self.get_images_from_path(slide_path)
        # get the slide label
        label = torch.from_numpy(self.labels[idx])
        
        sample = {'imgs': data_dict['imgs'],
                  'img_lens': data_dict['img_lens'],
                  'coords': data_dict['coords'],
                  'slide_id': slide_id,
                  'labels': label}
        return sample
    
    def get_sample_with_try(self, idx, n_try=3):
        '''Get the sample with n_try'''
        for _ in range(n_try):
            try:
                sample = self.get_one_sample(idx)
                return sample
            except:
                print('Error in getting the sample, try another index')
                idx = np.random.randint(0, len(self.slide_data))
        print('Error in getting the sample, skip the sample')
        return None
        
    def __len__(self):
        return len(self.slide_data)
    
    def __getitem__(self, idx):
        sample = self.get_sample_with_try(idx)
        return sample