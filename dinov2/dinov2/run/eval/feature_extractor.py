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

def eval_log_regression(
    *,
    model,
    train_dataset,
    batch_size,
    num_workers,
    train_features_device=_CPU_DEVICE,
    

):
    """
    Implements the "standard" process for log regression evaluation:
    The value of C is chosen by training on train_dataset and evaluating on
    finetune_dataset. Then, the final model is trained on a concatenation of
    train_dataset and finetune_dataset, and is evaluated on val_dataset.
    If there is no finetune_dataset, the value of C is the one that yields
    the best results on a random 10% subset of the train dataset
    """

    start = time.time()

    train_features, train_labels = extract_features(
        model, train_dataset, batch_size, num_workers, gather_on_cpu=(train_features_device == _CPU_DEVICE)
    )
    #save train_features
    return train_features


def get_log_regression_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--finetune-dataset-str",
        dest="finetune_dataset_str",
        type=str,
        help="Fine-tuning dataset",
    )
    parser.add_argument(
        "--finetune-on-val",
        action="store_true",
        help="If there is no finetune dataset, whether to choose the "
        "hyperparameters on the val set instead of 10%% of the train dataset",
    )
    parser.add_argument(
        "--metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Metric type",
    )
    parser.add_argument(
        "--train-features-device",
        type=str,
        help="Device to gather train features (cpu, cuda, cuda:0, etc.), default: %(default)s",
    )
    parser.add_argument(
        "--train-dtype",
        type=str,
        help="Data type to convert the train features to (default: %(default)s)",
    )
    parser.add_argument(
        "--max-train-iters",
        type=int,
        help="Maximum number of train iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        help="Number of GPUs to request on each node",
    )
    parser.add_argument(
        "--cluster_num",
        type=int,
        help="Number of clusters"
    )
    parser.add_argument(
        "--dump_path",
        type=str,
        help="Path to save the features",
    )
    parser.add_argument('--dataset_list',
                        nargs='+', 
                        type=str, 
                        help='Dataset for feature extraction')
    
    
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        finetune_dataset_str=None,
        metric_type=MetricType.MEAN_ACCURACY,
        train_features_device="cpu",
        train_dtype="float64",
        finetune_on_val=False,
        cluster_num=50,
        dump_path="/ruiyan/yuhao/embedding",
        ngpus=4
    )
    
    
    
    return parser

def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model

def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)
    model = build_model_for_eval(config, args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype

def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float
    
def eval_feature_extraction_with_model(
    model,
    train_dataset_str="ImageNet:split=TRAIN",
    autocast_dtype=torch.float,
    train_features_device=_CPU_DEVICE,
    dataset_list: list = [],
    dump_path: str = None,
):
    cudnn.benchmark = True

    transform = make_classification_eval_transform(resize_size=224)
    target_transform = None

    train_dataset = make_dataset(dataset_str=train_dataset_str, 
                                 transform=transform,
                                 target_transform=target_transform,
                                 dataset_list = dataset_list)


    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        train_feature=eval_log_regression(
            model=model,
            train_dataset=train_dataset,
            batch_size=8192,
            #batch_size=6144,
            num_workers=2,  
            train_features_device=train_features_device,
        )
    cluster_features(train_dataset.image_paths, train_feature, dump_path=dump_path)
    torch.distributed.barrier()

@torch.no_grad()
def cluster_features(train_json_data, train_feature, cluster_num=50, dump_path=""):
    patch_slide_list = [os.path.basename(os.path.dirname(x)) 
                        for x in train_json_data]
    patch_number_dict = collections.OrderedDict(Counter(patch_slide_list))
    batch_pos_list=[os.path.splitext(os.path.basename(x))[0] 
                    for x in train_json_data]

    patch_position_list=[tuple(re.sub('[a-zA-Z]','',name).split('_')) 
                            for name in batch_pos_list]
    patch_position_list =np.array([[int(i),int(j)] for i,j in patch_position_list])
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    start_index = 0
    for wsi_name, wsi_count in patch_number_dict.items():
        wsi_feature = train_feature[start_index:start_index + wsi_count]
        wsi_position = patch_position_list[start_index:start_index + wsi_count]
        start_index += wsi_count
        print(wsi_feature.shape)
        if wsi_feature.shape[0] < cluster_num:
            continue

        kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(wsi_feature.cpu())
        data_dict={
            'features':wsi_feature.cpu(),
            "img_len":[len(wsi_position)],
            'labels':kmeans.labels_,
            'pad_mask':[0],
            'coords':wsi_position,
            'cluster_centers':kmeans.cluster_centers_
        }
        with h5py.File(os.path.join(dump_path, wsi_name.split('.')[0] + '.h5'), 'w') as f:
            for key, value in data_dict.items():
                f.create_dataset(key, data=value)
        print(wsi_name,'features saved')
        
        
if __name__=="__main__":
    description = "feature_extractor"
    log_regression_args_parser = get_log_regression_args_parser(add_help=False)
    parents = [log_regression_args_parser]
    parser = argparse.ArgumentParser(description=description, parents=parents, add_help=False)
    args = parser.parse_args()
    model, autocast_dtype = setup_and_build_model(args)
    eval_feature_extraction_with_model(
        model=model,
        train_dataset_str=args.train_dataset_str,
        autocast_dtype=autocast_dtype,
        train_features_device=torch.device(args.train_features_device),
        dataset_list=args.dataset_lst,
        dump_path=args.dump_path
    )
    
