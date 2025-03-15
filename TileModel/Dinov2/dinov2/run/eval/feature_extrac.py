
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
sys.path.append('/home/yuhaowang/project/FMBC/TileModel/Dinov2')
from dinov2.data import make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils
import random
import logging
import torch
import logging 
from urllib.parse import urlparse
logger = logging.getLogger("dinov2")
def load_pretrained_weights(model, pretrained_weights, checkpoint_key=None):
    if urlparse(pretrained_weights).scheme:  # 如果是 URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cuda")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cuda")

    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key '{checkpoint_key}' from checkpoint")
        state_dict = state_dict[checkpoint_key]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Pretrained weights loaded from {pretrained_weights} with message: {msg}")
    
    return model
def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model
def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Extract features for a single dataset")
    parser.add_argument('--img_dir', type=str, default='/data4/processed_data', help='Directory containing datasets')
    parser.add_argument('--save_dir', type=str, default='/data4/embedding', help='Directory to save extracted features')
    parser.add_argument('--pretrained_weights', type=str, default='/home/yuhaowang/project/FMBC/TileModel/Dinov2/finetuning_399999.pth', help='Path to pretrained model weights')
    parser.add_argument('--batch_size', type=int, default=120, help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=160, help='Number of workers for data loading')
    parser.add_argument('--dataset_name', type=str,default='TCGA-BRCA', help='Single dataset name to process')
    parser.add_argument('--prefix_name', type=str, default='FMBC', help='Model name to use')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA GPU id to use')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # config = EasyDict({
    #     'student': EasyDict({
    #         'arch': 'vit_h',
    #         'patch_size': 14,
    #         'drop_path_rate': 0.4,
    #         'layerscale': 1e-4,
    #         'ffn_layer': 'SwiGLUPacked',
    #         'block_chunks': 4,
    #         'num_register_tokens': 8,
    #         'interpolate_offset': 0.1,
    #         'qkv_bias': True,
    #         'proj_bias': True,
    #         'ffn_bias': True,
    #         'interpolate_antialias': False,
    #         'interpolate_offset': 0.1
    #     }),
    #     'crops': EasyDict({
    #         'global_crops_size': 224,
    #     })    
    # })
    
    config = EasyDict({
        'student': EasyDict({
            'arch': 'vit_base',  # 覆盖默认值 'vit_large'
            'patch_size': 16,  # 保持不变
            'drop_path_rate': 0.3,  # 保持不变
            'layerscale': 1.0e-05,  # 保持不变
            'drop_path_uniform': True,  # 保持不变
            'pretrained_weights': '',  # 保持不变
            'ffn_layer': 'mlp',  # 保持不变
            'block_chunks': 4,  # 覆盖默认值 0
            'qkv_bias': True,  # 保持不变
            'proj_bias': True,  # 保持不变
            'ffn_bias': True,  # 保持不变
            'num_register_tokens': 0,  # 保持不变
            'interpolate_antialias': False,  # 保持不变
            'interpolate_offset': 0.1  # 保持不变
        }),
        'crops': EasyDict({
            'global_crops_size': 224,
        })    
    })


    model = build_model_for_eval(config, args.pretrained_weights)
    transform = make_classification_eval_transform(resize_size=224)

    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name, args.prefix_name)
    os.makedirs(dataset_save_dir, exist_ok=True)

    dataset_path = os.path.join(args.img_dir, args.dataset_name, 'output')
    if not os.path.exists(dataset_path):
        print(f"Warning: {dataset_path} does not exist, skipping...")
        return
    slide_list = os.listdir(dataset_path)
    random.shuffle(slide_list)
    for slide in tqdm(slide_list, desc=f"Processing {args.dataset_name}"):
        slide_name = slide.split('.')[0]
        save_path = os.path.join(dataset_save_dir, f"{slide_name}.h5")

        if os.path.exists(save_path):
            print(f"{slide} has been processed")
            continue

        try:
            dataset_str = f"TileDataset:split=VALID:root={dataset_path}/{slide}"
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

            with h5py.File(save_path, 'w') as f:
                f.create_dataset('features', data=feature.numpy())
                f.create_dataset('coords', data=coords.numpy())
            
            print(f"{slide} features saved successfully!")

        except Exception as e:
            print(f"Error processing {slide}: {e}")
            print(traceback.format_exc())
    
    print(f"Dataset {args.dataset_name} processed successfully!")

if __name__ == "__main__":
    main()
