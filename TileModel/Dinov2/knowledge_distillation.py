import os
import torch
import torch.optim as optim
import torch.nn as nn
import timm
from easydict import EasyDict
from dinov2.data import make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.models import build_model_from_cfg
import dinov2.utils.utils as dinov2_utils
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, random_split
import torch.distributed as dist
import os
import datasets
from datasets import Dataset, Image
import glob 

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def build_student(config, pretrained_weights=None):
    """构建 Student 模型"""
    # 从配置文件中构建模型，只构建 Teacher 模型
    model, _ = build_model_from_cfg(config, only_teacher=True)
    # 如果有预训练权重，则加载预训练权重
    if pretrained_weights is not None:
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    # 返回构建好的模型
    return model

# 配置 Student 模型
student_config = EasyDict({
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

def build_teacher():
    """构建 Teacher 模型"""
    local_dir = './'
    teacher_timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,  # = 5.33334
        'num_classes': 0,        
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    teacher_model = timm.create_model(pretrained=False, **teacher_timm_kwargs)
    teacher_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin")), strict=True)
    teacher_model.eval()
    return teacher_model

###################################
# 3. Dataset Preparation
###################################
transform = make_classification_eval_transform(resize_size=224)
def transforms(examples):
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset_str = "/data4/processed_data/SLN-Breast/output/HobI16-053768896760.svs"
#glob all png recursive
image_dir = glob.glob(os.path.join(dataset_str, "**/*.png"), recursive=True)

#remove all png contain thumbnail
image_dir = [x for x in image_dir if "thumbnail" not in x]
#split train and val
train_dir = image_dir[:int(0.8 * len(image_dir))]
val_dir = image_dir[int(0.8 * len(image_dir)):]
dataset_train = Dataset.from_dict({"image": train_dir}).cast_column("image", Image())
dataset_val = Dataset.from_dict({"image": val_dir}).cast_column("image", Image())

dataset_train.set_transform(transforms)
dataset_val.set_transform(transforms)
# full_dataset = make_dataset(dataset_str=dataset_str, transform=transform, target_transform=None)

student_model = build_student(student_config)
teacher_model = build_teacher()
for param in teacher_model.parameters():
    param.requires_grad = False  # Teacher 只做前向传播

#merge teacher and student as one knowledge distillation model

class knowledge_distillation(nn.Module):
    def __init__(self, teacher_model, student_model, temperature=5.0, lambda_param=0.5,):
        super(knowledge_distillation, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.lambda_param = lambda_param
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        
    def forward(self, x):
        # Student forward
        student_logits = self.student_model(x)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)

        # Distillation loss
        # 对教师模型的输出进行softmax操作，并除以温度参数
        soft_teacher = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1)

        soft_student = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        # 计算蒸馏损失，乘以温度参数的平方
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        return distillation_loss* self.lambda_param 
###################################
# 4. Custom Distillation Trainer
###################################
knowledge_distillation_model = knowledge_distillation(teacher_model, student_model)
class ImageDistilTrainer(Trainer):
    def __init__(self, model=None, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        
        # self.temperature = temperature
        # self.lambda_param = lambda_param

    def compute_loss(self, _, inputs, return_outputs=False, **kwargs):
        """
        Expecting inputs to have:
           - inputs["sample"][0] as the batch of images (pixel_values)
           - inputs["labels"] as the class labels
        Adjust if your dataset keys differ.
        """
        pixel_values = inputs["pixel_values"]  # from dataset, e.g. images


        distillation_loss = self.model(pixel_values)
        return distillation_loss
    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in eval_dataloader:
            batch = {k: v.to(self.args.device) for k, v in batch.items()}
            
            with torch.no_grad():
                loss = self.compute_loss('_',batch)
            
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Validation Loss: {avg_loss}")
        return {"val_loss": avg_loss}

###################################
# 5. Training Arguments
###################################

# 检查是否为多 GPU 训练
num_gpus = 1
is_distributed = num_gpus > 1 and dist.is_available()

training_args = TrainingArguments(
    output_dir="my-awesome-model",
    num_train_epochs=30,
    fp16=False, #if torch.cuda.is_available() else False,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    logging_dir="logs",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True, 
    metric_for_best_model="val_loss",  # pick best by val loss
    greater_is_better=False,
    push_to_hub=False,
    hub_strategy="every_save",
    hub_model_id="my-awesome-model",
    report_to="tensorboard",
    ddp_find_unused_parameters=True if is_distributed else None,
    ddp_backend="nccl" if is_distributed else None,
    remove_unused_columns=False

)
def collate_fn(examples):
    # 将每个样本的像素值堆叠在一起
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # 返回堆叠后的像素值
    return {"pixel_values": pixel_values}


if __name__ == "__main__":
    trainer = ImageDistilTrainer(
        model=knowledge_distillation_model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_fn
    )

    # 训练
    trainer.train()

    # 评估
    eval_results = trainer.evaluate()
    print("Final eval results:", eval_results)
