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
import random
from urllib.parse import urlparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import models.vision_transformer as vits
import numpy as np
from torch.utils.data import Dataset

def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        model.eval()
        return model

def read_assets_from_h5(h5_path: str) -> tuple:
    assets = {}
    attrs = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            assets[key] = f[key][:]
            if f[key].attrs is not None:
                attrs[key] = dict(f[key].attrs)
    return assets, attrs

class H5Dataset(Dataset):
    def __init__(self, directory: str, feature_key: str = 'features', coord_key: str = 'coords'):
        self.directory = directory
        self.feature_key = feature_key
        self.coord_key = coord_key
        self.h5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
        
        if not self.h5_files:
            raise ValueError("No h5 files found in the given directory.")

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.h5_files):
            raise IndexError("Index out of range.")
        
        h5_path = self.h5_files[idx]
        assets, _ = read_assets_from_h5(h5_path)
        
        if self.feature_key not in assets:
            raise KeyError(f"Key '{self.feature_key}' not found in {h5_path}")

        features = np.array(assets[self.feature_key])
        #some feature too long, max length setting 8000
        if features.shape[0] > 16000:
            #shuffle features
            np.random.shuffle(features)
            features = features[:16000]
        name = os.path.splitext(os.path.basename(h5_path))[0]
        return name,features

def parse_args():
    parser = argparse.ArgumentParser(description="Extract features for all H5 files in a directory")
    parser.add_argument('--h5_dir', type=str, default='/data4/embedding', help='Directory containing input H5 files')
    parser.add_argument('--save_dir', type=str,default='/data4/embedding' ,help='Directory to save extracted features')
    parser.add_argument('--pretrained_weights', type=str,default='/home/yuhaowang/project/FMBC/Weights/slide/train_from_our_FMBC/checkpoint0160.pth' ,  help='Path to pretrained model weights')
    parser.add_argument('--gpu', type=str, default='1', help='CUDA GPU id to use')
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model_arch = 'vit_base'
    input_dim = 768
    slide_encoder = vits.__dict__[model_arch](slide_embedding_size=input_dim, return_all_tokens=False)
    #load_pretrained_weights(slide_encoder, args.pretrained_weights, 'teacher')
    slide_encoder.cuda()
    slide_encoder.eval()
    for folder in os.listdir(args.h5_dir):
        try:
        #if folder contain private , continue
            if 'private' in folder:
                continue
            dataset_dir = os.path.join(args.h5_dir, folder,'FMBC')
            save_dir = os.path.join(args.save_dir, folder,'FMBC_Slide')
            if len(os.listdir(save_dir)) == len(os.listdir(dataset_dir)):
                continue
            os.makedirs(save_dir, exist_ok=True)
            dataset = H5Dataset(dataset_dir)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
            
            for idx, (name, features) in enumerate(tqdm(dataloader, desc='Processing H5 files')):
                features = features.cuda()
                with torch.no_grad():
                    masks = torch.zeros(features.shape[0], 
                            features.shape[1], 
                            dtype=torch.bool).to(device=features.device)
                    coords = torch.zeros(features.shape[0], 2).to(device=features.device)
                    feature_vector = slide_encoder(features,coords,masks).detach().cpu().numpy()
                
                coords = np.random.rand(1, 2)  
                save_path = os.path.join(save_dir, f'{name[0]}.h5')
                
                with h5py.File(save_path, 'w') as f:
                    f.create_dataset('features', data=feature_vector)
                    f.create_dataset('coords', data=coords)
                
                print(f"Saved: {save_path}")
        except:
            pass
       

if __name__ == "__main__":
    main()