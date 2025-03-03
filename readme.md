# Data Prepartion Pipeline
## scp baseline feature embedding, including gigapath,chief, etc .... from 24 Node
```rsync -avz --progress --partial yuhaowang@172.16.120.24:/data1/embedding /data4/```
## Feature Embedding Extraction
```cd /home/yuhaowang/project/FMBC/dinov2/dinov2/run/eval```
```python multi_gpu.py```
