#!/bin/bash

# 指定CUDA设备
CUDA_VISIBLE_DEVICES=6

# 主要参数
TASK_CFG_PATH="task_configs/mutation_5_gene.yaml"
DATASET_CSV="dataset_csv/mutation/LUAD-5-gene_TCGA.csv"
ROOT_PATH="/home/yuhaowang/data/embedding/TCGA-LUAD"
BLR=0.004
LAYER_DECAY=0.95
OPTIM_WD=0.05
DROPOUT=0.1
DROP_PATH_RATE=0.0
VAL_R=0.1
EPOCHS=10
INPUT_DIM=384
LATENT_DIM=384
FEAT_LAYER=11
WARMUP_EPOCHS=0
GC=32
MODEL_SELECT="last_epoch"
LR_SCHEDULER="cosine"
FOLDS=1
SAVE_DIR="outputs/LUAD-5-gene"
MAX_WSI_SIZE=250000
PRETRAINED="/home/yuhaowang/project/FMBC/dino_stage2/temp_output/output/checkpoint0180.pth"
MODEL_ARCH="vit_small"

# 运行命令
python main.py \
    --task_cfg_path "$TASK_CFG_PATH" \
    --dataset_csv "$DATASET_CSV" \
    --root_path "$ROOT_PATH" \
    --blr "$BLR" \
    --layer_decay "$LAYER_DECAY" \
    --optim_wd "$OPTIM_WD" \
    --dropout "$DROPOUT" \
    --drop_path_rate "$DROP_PATH_RATE" \
    --val_r "$VAL_R" \
    --epochs "$EPOCHS" \
    --input_dim "$INPUT_DIM" \
    --latent_dim "$LATENT_DIM" \
    --feat_layer "$FEAT_LAYER" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --gc "$GC" \
    --model_select "$MODEL_SELECT" \
    --lr_scheduler "$LR_SCHEDULER" \
    --folds "$FOLDS" \
    --save_dir "$SAVE_DIR" \
    --max_wsi_size "$MAX_WSI_SIZE" \
    --pretrained "$PRETRAINED" \
    --model_arch "$MODEL_ARCH"
