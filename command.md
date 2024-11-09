# validation the TCGA-LUAD 5 gene mutation
```
CUDA_VISIBLE_DEVICES=3 python main.py --task_cfg_path task_configs/mutation_5_gene.yaml --dataset_csv dataset_csv/mutation/LUAD-5-gene_TCGA.csv --root_path /home/yuhaowang/data/test/TCGA-LUAD-AllFeature --blr 0.002 --layer_decay 0.95 --optim_wd 0.05 --dropout 0.1 \
--drop_path_rate 0.0 --val_r 0.1 --epochs 10 --input_dim 384 --latent_dim 384 --feat_layer 11 --warmup_epochs 1 --gc 32 --model_select last_epoch --lr_scheduler cosine --folds 1 \
--save_dir outputs/LUAD-5-gene --max_wsi_size 250000 --pretrained /home/yuhaowang/project/FMBC/dino_stage2/output/checkpoint0180.pth --model_arch vit_small
```