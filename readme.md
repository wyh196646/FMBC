# Data Prepartion Pipeline
## scp baseline feature embedding, including gigapath,chief, etc .... from 24 Node
```rsync -avz --progress --partial yuhaowang@172.16.120.24:/data1/embedding /data4/```
## Feature Embedding Extraction
```cd /home/yuhaowang/project/FMBC/dinov2/dinov2/run/eval```
```python multi_gpu.py```

## scp the checkpoint 
```rsync -avz --progress --partial -e 'ssh -p 30215' root@172.16.120.7:/ruiyan/yuhao/project/FMBC/pretrain_weights/pytorch_model.bin  ./``` 


## scp the h5 file, feature embedding
```rsync -avz --progress --partial -e 'ssh -p 30215' root@172.16.120.7:/ruiyan/yuhao/embedding /data4/embedding/temp ```