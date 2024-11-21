# validation the TCGA-LUAD 5 gene mutation
```
CUDA_VISIBLE_DEVICES=0 python main.py --task_cfg_path task_configs/mutation_5_gene.yaml --dataset_csv dataset_csv/mutation/LUAD-5-gene_TCGA.csv --root_path /home/yuhaowang/data/embedding/TCGA-LUAD --blr 0.004 --layer_decay 0.95 --optim_wd 0.05 --dropout 0.1 \
--drop_path_rate 0.0 --val_r 0.1 --epochs 10 --input_dim 384 --latent_dim 384 --feat_layer 11 --warmup_epochs 0 --gc 32 --model_select last_epoch --lr_scheduler cosine --folds 1 \
--save_dir outputs/LUAD-5-gene --max_wsi_size 250000 --pretrained /home/yuhaowang/project/FMBC/dino_stage2/output/checkpoint0180.pth --model_arch vit_small
```

# Valid on TCGA-LUAD 6 gene mutation
```
CUDA_VISIBLE_DEVICES=3 python main.py --task_cfg_path task_configs/mutation_6_gene_brca.yaml --dataset_csv dataset_csv/mutation/BRCA-6-gene_TCGA.csv --root_path /home/yuhaowang/data/embedding/TCGA-BRCA --blr 0.002 --layer_decay 0.95 --optim_wd 0.05 --dropout 0.1 \
--drop_path_rate 0.0 --val_r 0.1 --epochs 30 --input_dim 384 --latent_dim 384 --feat_layer 11 --warmup_epochs 1 --gc 32 --model_select last_epoch --lr_scheduler cosine --folds 5 \
--save_dir outputs/BRCA-6-gene --max_wsi_size 250000 --pretrained /home/yuhaowang/project/FMBC/dino_stage2/output/checkpoint0180.pth --model_arch vit_small
```


## 跨采样（模态）数量重建有两种方式
1.重建时，将所有模态拼接起来，比如1000+2000=3000，然后进行重建
2.使用Transformer的decoder，但是tgt_embd使用原型，设置一个可学习的参数，prototype，然后与源模态生成目标模态，注意mask
3.使用nn.functional.interpolate的采样方法进行数值采样