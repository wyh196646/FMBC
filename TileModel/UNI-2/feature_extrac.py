import logging
import os
import torch
import h5py
import traceback
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download
import random
sys.path.append('/home/yuhaowang/project/FMBC/TileModel/Dinov2')
from dinov2.data import make_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features for a single dataset")
    parser.add_argument('--img_dir', type=str, default='/data4/processed_data', help='Directory containing datasets')
    parser.add_argument('--save_dir', type=str, default='/data4/embedding', help='Directory to save extracted features')
    parser.add_argument('--batch_size', type=int, default=120, help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading')
    parser.add_argument('--dataset_name', type=str, default='TCGA-BRCA', help='Single dataset name to process')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA GPU id to use')  
    parser.add_argument('--prefix_name',type=str, default='FMBC', help='Prefix name for saving features')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #login(token='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')  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    local_dir = "./"
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    
    hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir)
    timm_kwargs = {
                'model_name': 'vit_giant_patch14_224',
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin")), strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    model.cuda()


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


#python multi_gpu.py