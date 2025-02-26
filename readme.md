# Data Prepartion Pipeline
## scp baseline feature embedding, including gigapath,chief,etc...
```rsync -avz --progress --partial yuhaowang@172.16.120.24:/data1/embedding /data4/fm_embedding```
## Feature Embedding Extraction
```cd /home/yuhaowang/project/FMBC/dinov2/dinov2/run/eval```
```python multi_gpu.py```
