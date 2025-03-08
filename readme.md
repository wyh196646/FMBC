# Data Prepartion Pipeline
## scp baseline feature embedding, including gigapath,chief, etc .... from 24 Node
```rsync -avz --progress --partial yuhaowang@172.16.120.24:/data1/embedding /data4/```
## Feature Embedding Extraction
```cd /home/yuhaowang/project/FMBC/dinov2/dinov2/run/eval```
```python multi_gpu.py```

## scp the checkpoint 
```rsync -avz --progress --partial -e 'ssh -p 30215' root@172.16.120.7:/ruiyan/yuhao/project/FMBC/pretrain_weights/pytorch_model.bin  ./``` 


## scp the UNI-2 h5 file, feature embedding
```rsync -avz --progress --partial -e 'ssh -p 30215' root@172.16.120.7:/ruiyan/yuhao/embedding /data4/embedding/temp ```


# scp feature from 34 to A800-01

```rsync -avz --progress --partial yuhaowang@172.16.120.34:/data4/embedding/temp/embedding /data/embedding```


## scp patch from 21
```rsync -avz --progress --partial yuhaowang@172.16.120.21:/mnt/data/ruiyan/processed_data/AHSL /data2```