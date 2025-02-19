import logging
import os
import torch
import h5py
import traceback
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict

import sys
sys.path.append('/ruiyan/yuhao/project/FMBC/dinov2')
from dinov2.data import make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils

logger = logging.getLogger("dinov2")

def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Extract features for specific datasets")
    parser.add_argument('--test_dir', type=str, default='/ruiyan/yuhao/data', help='Directory containing datasets')
    parser.add_argument('--save_dir', type=str, default='/ruiyan/yuhao/embedding', help='Directory to save extracted features')
    parser.add_argument('--pretrained_weights', type=str, default='/ruiyan/yuhao/project/FMBC/finetuning_UNI-1.5.pth', help='Path to pretrained model weights')
    parser.add_argument('--batch_size', type=int, default=160, help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=36, help='Number of workers for data loading')
    parser.add_argument('--dataset_names', type=str, default=None, nargs='*', help='List of dataset names to process')
    return parser.parse_args()

def main():
    args = parse_args()

    config = EasyDict({
        'student': EasyDict({
            'arch': 'vit_h',
            'patch_size': 14,
            'drop_path_rate': 0.4,
            'layerscale': 1e-4,
            'ffn_layer': 'SwiGLUPacked',
            'block_chunks': 4,
            'num_register_tokens': 8,
            'interpolate_offset': 0.1,
            'qkv_bias': True,
            'proj_bias': True,
            'ffn_bias': True,
            'interpolate_antialias': False,
            'interpolate_offset': 0.1
        }),
        'crops': EasyDict({
            'global_crops_size': 224,
        })    
    })

    model = build_model_for_eval(config, args.pretrained_weights)
    
    dataset_list = os.listdir(args.test_dir)
    #exclude_dir = [f'private_chunk_{i}' for i in range(1, 11)]
    exclude_dir = []
    dataset_list = [i for i in dataset_list if i not in exclude_dir]
    
    if args.dataset_names:
        dataset_list = [d for d in dataset_list if d in args.dataset_names]
    
    transform = make_classification_eval_transform(resize_size=224)

    for dataset in dataset_list:
        dataset_save_dir = os.path.join(args.save_dir, dataset, 'FMBC')
        os.makedirs(dataset_save_dir, exist_ok=True)

        dataset_path = os.path.join(args.test_dir, dataset, 'output')
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} does not exist, skipping...")
            continue

        for slide in tqdm(os.listdir(dataset_path), desc=f"Processing {dataset}"):
            slide_name = slide.split('.')[0]
            save_path = os.path.join(dataset_save_dir, f"{slide_name}.h5")

            if os.path.exists(save_path):
                print(f"{slide} has been processed")
                continue

            try:
                dataset_str = f"TileDataset:split=VALID:root={args.test_dir}/{dataset}/output/{slide}"
                train_dataset = make_dataset(dataset_str=dataset_str, transform=transform, target_transform=None)
                tile_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

                collated_outputs = {'tile_embeds': [], 'coords': []}

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    for batch in tqdm(tile_dl, desc='Running inference'):
                        sample = batch['sample'][0].cuda()
                        coords = torch.stack(batch['coords'], dim=1)

                        collated_outputs['tile_embeds'].append(model(sample).detach().cpu())
                        collated_outputs['coords'].append(coords.cpu())

                feature = torch.cat(collated_outputs['tile_embeds'])
                coords = torch.cat(collated_outputs['coords'])

                data_dict = {
                    'features': feature.numpy(),
                    'coords': coords.numpy(),
                }

                with h5py.File(save_path, 'w') as f:
                    for key, value in data_dict.items():
                        f.create_dataset(key, data=value)
                
                print(f"{slide} features saved successfully!")

            except Exception as e:
                print(f"Error processing {slide}: {e}")
                print(traceback.format_exc())

    print("All datasets processed successfully!")

if __name__ == "__main__":
    main()
