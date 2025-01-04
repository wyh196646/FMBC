import argparse
import gc
import logging
import sys
import time

import argparse
import gc
import logging
import sys
import time
from typing import List, Optional


import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import collections


import pandas as pd
import numpy as np
import json
import os
from PIL import Image, ImageFile
import collections
from collections import Counter
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import json
import torch
import re
import torch
import torch.backends.cudnn as cudnn
import torch.distributed
from torch import nn
import sys
sys.path.append('/ruiyan/yuhao/project/FMBC/dinov2')

from dinov2.data import make_dataset
from dinov2.data.transforms import make_classification_eval_transform

from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import evaluate, extract_features
from dinov2.utils.dtype import as_torch_dtype
from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup
from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup
import argparse
from typing import Any, List, Optional, Tuple
import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn


import dinov2.utils.utils as dinov2_utils
import torch
import torch.backends.cudnn as cudnn
import dinov2.utils.utils as dinov2_utils
import logging
logger = logging.getLogger("dinov2")
DEFAULT_MAX_ITER = 1_000
C_POWER_RANGE = torch.linspace(-6, 5, 45)
_CPU_DEVICE = torch.device("cpu")


def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model
from easydict import EasyDict
config = {
    'student':EasyDict({
    'arch': 'vit_base',
    'patch_size': 16,
    'drop_path_rate': 0.3,
    'layerscale': 1.0e-05,
    'drop_path_uniform': True,
    'pretrained_weights': '',
    'ffn_layer': 'mlp',
    'block_chunks': 4,
    'qkv_bias': True,
    'proj_bias': True,
    'ffn_bias': True,
    'num_register_tokens': 0,
    'interpolate_antialias': False,
    'interpolate_offset': 0.1
    }),
    'crops': 
        EasyDict({
            'global_crops_size': 224,
        })    
    
}

pretrained_weights='/ruiyan/yuhao/project/output/eval/training_499999/teacher_checkpoint.pth'
config=EasyDict(config)
model=build_model_for_eval(config, pretrained_weights)
from torch.utils.data import DataLoader
from tqdm import tqdm
test_dir = '/ruiyan/yuhao/data'
save_dir= '/ruiyan/yuhao/embedding'
#dataset_list = ['private_chunk_1','private_chunk_2','private_chunk_3']
dataset_list = ['TCGA-BRCA','private_chunk_4','private_chunk_5']

transform = make_classification_eval_transform(resize_size=224)
target_transform = None


batch_size = 640
for dataset in dataset_list:
    if not os.path.exists(os.path.join(save_dir,dataset)):
        os.makedirs(os.path.join(save_dir,dataset))
        
    for slide in tqdm(os.listdir(os.path.join(test_dir,dataset,'output'))):
        if os.path.exists(os.path.join(save_dir,dataset, slide.split('.')[0] + '.h5')):
            print(slide,'has been processed')
            continue
        try:
            train_dataset = make_dataset(dataset_str='TileDataset:split=VALID:root=/ruiyan/yuhao/data/{}/output/{}'.format(dataset, slide), 
                                        transform=transform,
                                        target_transform=target_transform,)

            tile_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
            # run inference
            collated_outputs = {'tile_embeds': [], 'coords': []}
            with torch.cuda.amp.autocast(dtype=torch.float16):
                for batch in tqdm(tile_dl, desc='Running inference with tile encoder'):
                    
                    collated_outputs['tile_embeds'].append(model(batch['sample'][0].cuda()).detach().cpu())
                    #print(batch['coords'])
                    collated_outputs['coords'].append( torch.stack(batch['coords'], dim=1))
                feature = torch.cat(collated_outputs['tile_embeds'])
                coords = torch.cat([torch.tensor(item) for item in collated_outputs['coords']])
                
                data_dict={
                    'features':feature.cpu(),
                    'coords':coords,
                }
                with h5py.File(os.path.join(save_dir,dataset, slide.split('.')[0] + '.h5'), 'w') as f:
                    print(os.path.join(os.path.join(save_dir,dataset), slide.split('.')[0] + '.h5'))
                    for key, value in data_dict.items():
                        f.create_dataset(key, data=value)
                print(slide,'features saved')
        except:
            print(slide,'failed')
    print(dataset,' done')
