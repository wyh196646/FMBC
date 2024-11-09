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
import utils
import vision_transformer as vits
import pandas as pd
import numpy as np
import json
import os
from PIL import ImageFile
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

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"
local_rank = int(os.environ["LOCAL_RANK"])
def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(args.data_path, transform=transform)
    
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    #print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    train_labels = list([s[0] for s in dataset_train.samples])


    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
    if args.dump_features and dist.get_rank() == 0:
        cluster_features(train_labels,train_features,args)
    print("all features have been extracted")
    return train_features, train_labels


@torch.no_grad()
def cluster_features(train_json_data, train_feature,args):
    patch_slide_list = [os.path.basename(os.path.dirname(x)) for x in train_json_data]
    patch_number_dict = collections.OrderedDict(Counter(patch_slide_list))
    patch_name_list = [os.path.splitext(os.path.basename(x))[0] for x in train_json_data]
    patch_position_list = [
        tuple(map(int, name.split('_', 1)[1].split('_'))) for name in patch_name_list
    ]
    start_index = 0
    for wsi_name, wsi_count in patch_number_dict.items():
        wsi_feature = train_feature[start_index:start_index + wsi_count]
        wsi_position = patch_position_list[start_index:start_index + wsi_count]
        start_index += wsi_count
        print(wsi_feature.shape)
        if wsi_feature.shape[0] < args.cluster_num:
            continue
        
        kmeans = KMeans(n_clusters=args.cluster_num, random_state=0).fit(wsi_feature.cpu())
        data_dict={
            'features':wsi_feature.cpu(),
            "img_len":[len(wsi_position)],
            'labels':kmeans.labels_,
            'pad_mask':[0],
            'coords':wsi_position,
            'cluster_centers':kmeans.cluster_centers_
        }
        with h5py.File(os.path.join(args.dump_features, wsi_name + '.h5'), 'w') as f:
            for key, value in data_dict.items():
                f.create_dataset(key, data=value)
        print(wsi_name,'features saved')
    
 
@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        #print(samples.shape)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features



class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=1,type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='/home/yuhaowang/project/FMBC/trash/WSI_preprocessed_code/feature_extractor/checkpoint0030.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--archHW', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default='/home/yuhaowang/data/embedding',
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--data_path', default='/home/yuhaowang/data/raw_data/TCGA-BRCA-Patch', type=str)
    parser.add_argument('--cluster_num',default=50 ,type=int, help='Number of cluster')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    if not os.path.exists(args.dump_features):
        os.makedirs(args.dump_features,exist_ok=True)
    


    train_features, train_labels = extract_feature_pipeline(args)

    dist.barrier()

# nohup torchrun --nproc_per_node=6 get_features.py &
# TCGA-LUAD 387G