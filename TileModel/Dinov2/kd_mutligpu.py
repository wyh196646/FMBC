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
from pytorch_lightning.callbacks import ModelCheckpoint


def build_student(config, pretrained_weights=None):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if pretrained_weights:
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    return model
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
def clean_dataset(dataset):
    return [sample for sample in dataset if sample is not None]

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

class KnowledgeDistillationDataset(Dataset):
    def __init__(self, image_dir, target_embedding_root, transform=None):
        self.dataset = find_png_files_parallel(image_dir, 32)
        self.transform = transform
        self.target_embedding_root = target_embedding_root
        self.dataset = [x for x in self.dataset if "thumbnail" not in x]

    def read_assets_from_h5(self, h5_path):
        assets, attrs = {}, {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                assets[key] = f[key][:]
                if f[key].attrs:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs

    def __len__(self):
        return len(self.dataset)
    
    def  generate_fallback_data(self):
        dummy_img = Image.new("RGB", (224, 224), (128, 128, 128))  # 灰色填充
        dummy_feature = torch.zeros(1536)  # 用 0 填充特征
        return dummy_img, dummy_feature
    
    def __getitem__(self, idx):
        image_path = self.dataset[idx]
        try:
            img = Image.open(image_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except (OSError, IOError) as e:
            print(f"警告: 无法打开图片 {image_path}, 使用默认数据. 错误信息: {e}")
            return self.generate_fallback_data()

        dataset_name = image_path.split('/')[-4]
        slide_name = image_path.split('/')[-2]
        target_h5 = os.path.join(self.target_embedding_root, dataset_name, 'UNI-2', slide_name.split('.')[0] + '.h5')
        extract_coords = np.array([int(num) for num in re.findall(r'\d+', image_path.split('/')[-1])])
        try:
            assets, attrs = self.read_assets_from_h5(target_h5)
            distances = np.linalg.norm(assets['coords'] - extract_coords, axis=1)
            index = np.argmin(distances)
            return img, torch.tensor(assets['features'][index], dtype=torch.float32)
        except:
            print(f"警告: H5 文件 {target_h5} 无效或缺失，使用默认数据.")
            return self.generate_fallback_data()



# 解析命令行参数
parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
parser.add_argument("--debug", action="store_true", help="Run in debug mode (single GPU, small batch size)")
parser.add_argument("--multi-gpu", action="store_true", help="Run with multiple GPUs using DDP")
args = parser.parse_args()

if args.debug:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(torch.cuda.device_count())))

dataset_path = "/data4/processed_data/TCGA-BRCA"
embedding_root = "/data4/embedding/temp/embedding"
dataset = KnowledgeDistillationDataset(dataset_path, embedding_root, transform=transform)
train_dataset, val_dataset = random_split(dataset, [int(0.95 * len(dataset)), len(dataset) - int(0.95 * len(dataset))])
batch_size = 80
num_workers = 16

# 定义一个函数，用于将batch中的None元素去除，并使用torch.utils.data.default_collate函数对batch进行合并
def collate_fn(batch):
    # 将batch中的None元素去除
    batch = [item for item in batch if item is not None]
    # 如果batch为空，则返回None
    if len(batch) == 0:
        return None
    # 使用torch.utils.data.default_collate函数对batch进行合并
    return torch.utils.data.default_collate(batch)

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, drop_last=True, collate_fn=collate_fn)


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
        'drop_path_uniform': True,
    }),
    "crops": EasyDict({"global_crops_size": 224}),
})


student_model = build_student(student_config)

wandb.init(project="knowledge_distillation", name="training_run")
logger = WandbLogger(project="knowledge_distillation", name="training_run")

class KnowledgeModel(pl.LightningModule):
    def __init__(self, net, lr, temperature=4.0):
        super().__init__()
        self.net = net
        self.lr = lr
        self.temperature = temperature
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")
        for name, param in self.net.named_parameters():
            print(name, param.requires_grad)


    def forward(self, x):
     
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = F.log_softmax(self(x) / self.temperature, dim=-1)
        y_soft = F.softmax(y / self.temperature, dim=-1)
        loss = self.loss_fn(y_hat, y_soft) * (self.temperature ** 2)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None or len(batch) == 0:
            return None  # 跳过非法数据
        x, y = batch
        y_hat = F.log_softmax(self(x) / self.temperature, dim=-1)
        y_soft = F.softmax(y / self.temperature, dim=-1)
        loss = self.loss_fn(y_hat, y_soft) * (self.temperature ** 2)
        # Explicitly log val_loss
        self.log("val_loss", loss, prog_bar=True,sync_dist=True,logger=True)
        return {"val_loss": loss}  # Ensure it returns loss for monitoring


    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs if x is not None]).mean()
    #     print(f"Validation Loss: {avg_loss.item()}")  # Debugging
    #     self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # 定义优化器，使用Adam优化算法，优化self.net的参数，学习率为self.lr
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # 定义学习率调度器，使用余弦退火算法，优化器为optimizer，最大周期为10，最小学习率为eta_min
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        # 返回优化器和调度器
        return [optimizer], [scheduler]

learning_rate = 1e-3
model = KnowledgeModel(student_model, learning_rate)

checkpoint_dir = "/home/yuhaowang/project/FMBC/TileModel/Dinov2/knowledge_ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


best_model_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="best_model",
    monitor="train_loss",
    mode="min",
    save_top_k=1,
    save_weights_only=True
)



trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    devices=1 if args.debug else torch.cuda.device_count(),
    strategy='ddp_find_unused_parameters_true',
    max_epochs=100,
    callbacks=[best_model_callback],  
    log_every_n_steps=1,  # Logs every step
    check_val_every_n_epoch=1,  # Ensure validation runs every epoch
)


trainer.fit(model, train_dl, val_dl)

print(f"Model checkpoints are saved in: {checkpoint_dir}")
print(f"Best model saved as: {best_model_callback.best_model_path}")
print("Training finished")


#  could not find the monitored key in the returned metrics: ['train_loss', 'epoch', 'step']. HINT: Did you call `log('val_loss', value)` in the `LightningModule`?
# Epoch 7:   0%|                                                                                                   | 0/2 [00:00<?, ?it/s, v_num=j8n5, train_loss=0.187