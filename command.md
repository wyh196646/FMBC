# Training:train patch encoder
'''
source activate dinov2 && cd /ruiyan/yuhao/project/FMBC/dinov2 && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port 11129  --nproc_per_node=6 dinov2/train/train.py --config-file=dinov2/configs/train/patch.yaml --output-dir=./output/ train.dataset_path=TileDataset:split=TRAIN:root=/ruiyan/yuhao/data
'''

# run patch feature extract

source activate dinov2 && cd /ruiyan/yuhao/project/FMBC/dinov2 && python dinov2/run/eval/feature_extractor.py  --config-file dinov2/configs/train/patch.yaml --pretrained-weights /ruiyan/yuhao/project/output/eval/training_499999/teacher_checkpoint.pth --output-dir ./ --train-dataset TileDataset:split=TRAIN:root=/ruiyan/yuhao/data --dump_path /ruiyan/yuhao/embedding/TCGA-BRCA --dataset_list TCGA-BRCA
 

# Downstreamtask 1: Valid on TCGA-BRCA 6 gene mutation
```

CUDA_VISIBLE_DEVICES=0 python main.py --task_cfg_path task_configs/mutation_6_gene_brca.yaml --dataset_csv dataset_csv/mutation/BRCA-6-gene_TCGA.csv --root_path /home/yuhaowang/data/embedding/TCGA-BRCA --blr 0.002 --layer_decay 0.95 --optim_wd 0.05 --dropout 0.1 --drop_path_rate 0.0 --val_r 0.1 --epochs 30 --input_dim 384 --latent_dim 384 --feat_layer 11 --warmup_epochs 0 --gc 32 --model_select last_epoch --lr_scheduler cosine --folds 5 --save_dir outputs --max_wsi_size 250000 --pretrained /home/yuhaowang/project/FMBC/dino_stage2/output/checkpoint0160.pth --model_arch vit_base 
```

# Downstreamtask 2:  Valid on TCGA-BRCA 6 gene mutation
```

CUDA_VISIBLE_DEVICES=6 python main.py --task_cfg_path task_configs/tcga-brca-gene_expression.yaml --dataset_csv dataset_csv/expression_prediction/COUNT_Symbol_matrix_transpose_slide_id.csv --root_path /home/yuhaowang/data/embedding/TCGA-BRCA --blr 0.002 --layer_decay 0.95 --optim_wd 0.05 --dropout 0.1 --drop_path_rate 0.0 --val_r 0.1 --epochs 30 --input_dim 384 --latent_dim 384 --feat_layer 11 --warmup_epochs 0 --gc 32 --model_select last_epoch --lr_scheduler cosine --folds 5 --save_dir outputs --max_wsi_size 250000 --pretrained /home/yuhaowang/project/FMBC/dino_stage2/output/checkpoint0160.pth --model_arch vit_base 
```
# Downstreamtask 3:  Valid on BRACS-Coarse 
```
CUDA_VISIBLE_DEVICES=0 python main.py --task_cfg_path task_configs/bracs_coarse.yaml --dataset_csv dataset_csv/subtype/BRACS_coarse.csv --root_path /ruiyan/yuhao/embedding/BRACS 

```


# Downstreamtask 3:  Valid on BRACS-Fine-Grained
```
CUDA_VISIBLE_DEVICES=0 python main.py --task_cfg_path task_configs/bracs_fine.yaml --dataset_csv dataset_csv/subtype/BRACS_fine.csv --root_path /ruiyan/yuhao/embedding/BRACS 

```


### Downstreamtask 3:  Valid on TCGA-Subtype
```
CUDA_VISIBLE_DEVICES=0 python main.py --task_cfg_path task_configs/TCGA-BRCA-Subtype.yaml --dataset_csv dataset_csv/subtype/TCGA-BRCA-Subtype.csv --root_path /ruiyan/yuhao/embedding/TCGA-BRCA 

```


### Downstreamtask 3:  Valid on bcnb_er
```
CUDA_VISIBLE_DEVICES=0 python main.py --task_cfg_path task_configs/bcnb_er.yaml --dataset_csv dataset_csv/biomarker/BCNB_ER.csv --root_path /ruiyan/yuhao/embedding/BCNB 

```

###  Downstreamtask 3:  Valid on bcnb_her2
```
CUDA_VISIBLE_DEVICES=0 python main.py --task_cfg_path task_configs/bcnb_her2.yaml --dataset_csv dataset_csv/biomarker/BCNB_HER2.csv --root_path /ruiyan/yuhao/embedding/BCNB 

```
# Downstreamtask 3:  Valid on bcnb_pr
```
CUDA_VISIBLE_DEVICES=0 python main.py --task_cfg_path task_configs/bcnb_pr.yaml --dataset_csv dataset_csv/biomarker/BCNB_PR.csv --root_path /ruiyan/yuhao/embedding/BCNB 

```
# Downstreamtask 3:  Valid on bcnb_aln
```
CUDA_VISIBLE_DEVICES=0 python main.py --task_cfg_path task_configs/bcnb_aln.yaml --dataset_csv dataset_csv/subtype/BCNB_ALN3subtype.csv --root_path /ruiyan/yuhao/embedding/BCNB --input_dim 768 --latent_dim 768 
```


# Downstreamtask 3:  TCGA-Gene-expression
```
CUDA_VISIBLE_DEVICES=0 python main.py --task_cfg_path task_configs/TCGA-BRCA-Gene-Exp.yaml --dataset_csv dataset_csv/expression_prediction/TCGA-Genexp.csv --root_path /ruiyan/yuhao/embedding/TCGA-BRCA --input_dim 768 --latent_dim 768 
```