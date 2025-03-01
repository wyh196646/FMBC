# Classification / Regression / Survival Analysis Run with python main.py
```python parallel.py```
# Retrieval Tasks Run / May be some other tasks： radiology report generation etc..., with patho_bench or


# 尝试
目前有一个基本的结论，
1. finetuning就是带着slide encoder的参数一起微调
2. Retrival 任务是没有模型参数的，仅用计算出来的特征
3. Logistic Regression 是仅通过特征 + sklearn里的logistic regression进行回归的发， 几本就是用一个 1*768 的向量进行逻辑回归
 