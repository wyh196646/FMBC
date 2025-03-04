from transformers import AutoModelForImageClassification, MobileNetV2Config, MobileNetV2ForImageClassification
from huggingface_hub import notebook_login
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoImageProcessor
import evaluate
import numpy as np
import timm
from torchvision import transforms
from transformers import DefaultDataCollator
import os
import sys
sys.path.append('/home/yuhaowang/project/FMBC/TileModel')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dinov2.data import make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils
import random
from easydict import EasyDict

transform = make_classification_eval_transform(resize_size=224)

def build_student(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    if pretrained_weights is not None:
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.cuda()
    return model



studen_config = EasyDict({
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
student_model = build_student(studen_config) 
 
local_dir = './'
teacher_timm_kwargs = {
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
teacher_model = timm.create_model(
    pretrained=False, **teacher_timm_kwargs
)
teacher_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin")), strict=True)
teacher_model.eval()
teacher_model.cuda()



teacher_model.eval()  # 设置为评估模式
for param in teacher_model.parameters():
    param.requires_grad = False  # 不更新教师模型权重

# 初始化学生模型 (MobileNetV2)

student_model = MobileNetV2ForImageClassification(student_config).to(device)

# 计算指标
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
    return {"accuracy": acc["accuracy"]}


data_collator = DefaultDataCollator()

# 训练参数 (启用多 GPU)
training_args = TrainingArguments(
    output_dir="my-awesome-model",
    num_train_epochs=30,
    fp16=True,  # 启用混合精度训练 (AMP)
    per_device_train_batch_size=16,  # 根据 GPU 调整 batch size
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    logging_dir="logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,  # 只保留最近两个模型
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id="my-awesome-model",
    ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,  # 多 GPU 模式
)

# 自定义蒸馏训练器
class ImageDistilTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, temperature=5, lambda_param=0.5, *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, student, inputs, return_outputs=False):
        student_output = self.student(**inputs)

        with torch.no_grad():
            teacher_output = self.teacher(**inputs)

        # 计算蒸馏损失
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # 计算分类损失
        student_target_loss = student_output.loss

        # 计算总损失
        loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss

# 登录 Hugging Face
notebook_login()

# 初始化训练器
trainer = ImageDistilTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    temperature=5,
    lambda_param=0.5
)

# 训练
trainer.train()

# 评估
trainer.evaluate(processed_datasets["test"])

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py
