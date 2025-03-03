# Classification / Regression / Survival Analysis Run with python main.py 
分类 回归 生存任务的线性微调就用我们这套框架 ， 使用parallel就可以了
# Retrieval Tasks Run / May be some other tasks： radiology report generation etc..., with patho_bench or
分类任务的 simple shot/ few shot 用PathoBench

# 尝试
目前有一个基本的结论，
1. finetuning就是带着slide encoder的参数一起微调
2. Retrival 任务是没有模型参数的，仅用计算出来的特征

3. Logistic Regression 是仅通过特征 + sklearn里的logistic regression进行回归的发， 几本就是用一个 1*768 的向量进行逻辑回归



## 注释
通常用0.0001 的学习率就可以了，但是CHIEF用0.001才能收敛，可以多尝试几个学习率看看，
看看0.001会不会对FMBC的微调有影响(正常来说FMBC应该用0.0001)
 