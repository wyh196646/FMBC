# Data Prepartion Pipeline
## scp baseline feature embedding, including gigapath,chief, etc .... from 24 Node
```rsync -avz --progress --partial yuhaowang@172.16.120.24:/data1/embedding /data4/```
## Feature Embedding Extraction
```cd /home/yuhaowang/project/FMBC/dinov2/dinov2/run/eval```
```python multi_gpu.py```

## scp the checkpoint 
```rsync -avz --progress --partial -e 'ssh -p 30215' root@172.16.120.7:/ruiyan/yuhao/project/FMBC/pretrain_weights/pytorch_model.bin  ./``` 


## scp the checkpoint from A800
```rsync -avz --progress --partial -e 'ssh -p 30215' root@172.16.120.7:/ruiyan/yuhao/project/stage1_weights/scratchtraining-500000iteration.pth ./ ```


## scp feature from 34 to A800-01


```rsync -avz --progress --partial yuhaowang@172.16.120.34:/home/yuhaowang/project/FMBC/downstream/finetune/outputs /home/yuhaowang/project/FMBC/downstream/finetune`

## scp patch from 21
```rsync -avz --progress --partial yuhaowang@172.16.120.21:/mnt/data/ruiyan/processed_data/AHSL /data2```
## only scp the FMBC feature command, not including other Foundation Models Features
```  
rsync -av --include='*/' --include='*/FMBC/**' --exclude='*' yuhaowang@172.16.120.62:/data4/embedding/ /data1/yuhaowang
   
```

# Feature Embedding Extraction
## Tile Level Feature Extraction
```cd /home/yuhaowang/project/FMBC/TileModel/Dinov2/dinov2/run/eval/multi_gpu.py```
```python multi_gpu.py```
