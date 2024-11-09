# from datasets import load_dataset

# dataset = load_dataset("imagefolder", data_dir="/home/yuhaowang/data/processed_data/TCGA-LUAD")
import torchvision
from torchvision.datasets import ImageFolder
dataset= ImageFolder("/home/yuhaowang/data/processed_data/TCGA-BRCA/output")
from transformers import Trainer, TrainingArguments
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import os
# from datasets import load_dataset

# dataset = load_dataset("imagefolder", data_dir="/home/yuhaowang/data/processed_data/TCGA-LUAD")
import torchvision
from torchvision.datasets import ImageFolder
from datasets import Dataset
#dataset= ImageFolder("/home/yuhaowang/data/processed_data/TCGA-BRCA/output")

class TileDataset(ImageFolder):
    def __init__(self, root, transform=None, loader=Image.open):
        super(TileDataset, self).__init__(root, transform=transform,loader=loader)
        self.samples = [(os.path.join(root, x[0]), x[1]) for x in self.samples]

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:  
            sample = self.transform(sample)
        result = {"pixel_values": sample}
        return result

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = TileDataset("/home/yuhaowang/data/processed_data/TCGA-BRCA/output", transform=transform)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = AutoModel.from_pretrained('facebook/dinov2-base')

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset,         # training dataset
    eval_dataset=dataset             # evaluation dataset
)

trainer.train()