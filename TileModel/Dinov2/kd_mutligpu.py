import os
import torch
import argparse
import numpy as np
import glob
import h5py
import re
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import wandb  # 添加 wandb
from pytorch_lightning.loggers import WandbLogger  # WandbLogger
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from easydict import EasyDict
from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils
import multiprocessing
import torch.nn.functional as F

def find_png_files(root_dir):
    """使用 os.scandir() 递归查找 PNG 文件"""
    png_files = []
    stack = [root_dir]
    while stack:
        current_dir = stack.pop()
        try:
            with os.scandir(current_dir) as entries:
                for entry in entries:
                    if entry.is_dir():
                        stack.append(entry.path)
                    elif entry.is_file() and entry.name.endswith(".png"):
                        png_files.append(entry.path)
        except PermissionError:
            continue
    return png_files

def find_png_files_parallel(root_dir, num_workers=8):
    if not os.path.isdir(root_dir):
        raise ValueError(f"Error: {root_dir} 不是一个目录")
    subdirs = [root_dir] + [
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(find_png_files, subdirs)
    png_files = [file for res in results for file in res]
    return png_files

# 解析命令行参数
parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
parser.add_argument("--debug", action="store_true", help="Run in debug mode (single GPU, small batch size)")
parser.add_argument("--multi-gpu", action="store_true", help="Run with multiple GPUs using DDP")
args = parser.parse_args()

if args.debug:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(torch.cuda.device_count())))

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

class KnowledgeDistillationDataset(Dataset):
    def __init__(self, image_dir, target_embedding_root, transform=None):
        self.dataset = find_png_files_parallel(image_dir, 8)
        self.transform = transform
        self.target_embedding_root = target_embedding_root
        self.dataset = [x for x in self.dataset if "thumbnail" not in x]

    def read_assets_from_h5(self, h5_path):
        assets, attrs = {}, {}
        try:
            with h5py.File(h5_path, 'r') as f:
                for key in f.keys():
                    assets[key] = f[key][:]
                    if f[key].attrs:
                        attrs[key] = dict(f[key].attrs)
        except (OSError, KeyError, IOError) as e:
            print(f"警告: 无法读取 H5 文件 {h5_path}, 错误信息: {e}")
            return None, None
        return assets, attrs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx]
        try:
            img = Image.open(image_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except (OSError, IOError) as e:
            print(f"警告: 无法打开图片 {image_path}, 错误信息: {e}")
            return None

        dataset_name = image_path.split('/')[-4]
        slide_name = image_path.split('/')[-2]
        target_h5 = os.path.join(self.target_embedding_root, dataset_name, 'UNI-2', slide_name.split('.')[0] + '.h5')
        extract_coords = np.array([int(num) for num in re.findall(r'\d+', image_path.split('/')[-1])])
        assets, attrs = self.read_assets_from_h5(target_h5)
        if assets is None or "coords" not in assets or "features" not in assets:
            return None
        distances = np.linalg.norm(assets['coords'] - extract_coords, axis=1)
        index = np.argmin(distances)
        return img, torch.tensor(assets['features'][index], dtype=torch.float32)

def clean_dataset(dataset):
    return [sample for sample in dataset if sample is not None]

dataset_path = "/data4/processed_data/"
embedding_root = "/data4/embedding/temp/embedding"
dataset = KnowledgeDistillationDataset(dataset_path, embedding_root, transform=transform)
train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
batch_size = 32 if args.debug else 80
num_workers = 4 if args.debug else 12
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

student_config = EasyDict({
    "student": EasyDict({
        "arch": "vit_h",
        "patch_size": 14,
        "drop_path_rate": 0.4,
        "layerscale": 1e-4,
        "ffn_layer": "SwiGLUPacked",
        "block_chunks": 4,
        "num_register_tokens": 8,
        "interpolate_offset": 0.1,
        "qkv_bias": True,
        "proj_bias": True,
        "ffn_bias": True,
        "interpolate_antialias": False,
        "interpolate_offset": 0.1,
    }),
    "crops": EasyDict({"global_crops_size": 224}),
})
def build_student(config, pretrained_weights=None):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if pretrained_weights:
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    return model

student_model = build_student(student_config)

wandb.init(project="knowledge_distillation", name="training_run")
logger = WandbLogger()

class KnowledgeModel(pl.LightningModule):
    def __init__(self, net, lr, temperature=4.0):
        super().__init__()
        self.net = net
        self.lr = lr
        self.temperature = temperature
        self.loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = F.log_softmax(y_hat / self.temperature, dim=-1)
        y = F.softmax(y / self.temperature, dim=-1)
        loss = self.loss(y_hat, y) * (self.temperature ** 2)
        self.log("train_loss", loss)
        wandb.log({"train_loss": loss.item()})
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        y_hat = F.log_softmax(y_hat / self.temperature, dim=-1)
        y = F.softmax(y / self.temperature, dim=-1)

        loss = self.loss(y_hat, y) * (self.temperature ** 2)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        return [optimizer], [scheduler]

model = KnowledgeModel(build_model_from_cfg(EasyDict({"arch": "vit_h"})), lr=1e-3)
trainer = pl.Trainer(logger=logger,
                     accelerator="gpu",
                     devices=1 if args.debug else torch.cuda.device_count(), 
                     strategy='ddp_find_unused_parameters_true',
                     max_epochs=2)
wandb.watch(model)
trainer.fit(model, train_dl, val_dl)
