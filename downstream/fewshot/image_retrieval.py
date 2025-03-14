import os
import h5py
import torch
import pandas as pd
import numpy as np
import torch
import logging
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class H5Dataset():
    def __init__(self, feature_dirs, label_csv):
        self.feature_dirs = feature_dirs
        self.features, self.labels, self.slide_ids = self.load_data(feature_dirs, label_csv)
        
    def read_assets_from_h5(self, h5_path: str):
        """Read the assets from the h5 file"""
        assets = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                assets[key] = f[key][:]
        return assets
    
    def load_data(self, feature_dirs, label_csv):
        label_df = pd.read_csv(label_csv)
        
        # 过滤掉缺失的h5文件
        valid_slide_ids = [file for file in label_df['slide_id'] if file+'.h5' in os.listdir(feature_dirs)]
        label_df = label_df[label_df['slide_id'].isin(valid_slide_ids)]
        
        labels = label_df['label'].values  
        slide_ids = label_df['slide_id'].values  # 记录slide_id
        
        label_dict = {label: num for num, label in enumerate(np.unique(labels))}
        target_labels = np.array([label_dict[label] for label in labels])
        
        all_features = []
        for file in slide_ids:
            h5_path = os.path.join(feature_dirs, file + '.h5')
            assets = self.read_assets_from_h5(h5_path)
            feature = assets['features']
   
            # mean pooling
            feature = np.mean(feature, axis=0) if feature.shape[0] > 1 else feature.flatten()
            all_features.append(feature)
    
        all_features = np.array(all_features)
        
        return torch.tensor(all_features, dtype=torch.float32), torch.tensor(target_labels, dtype=torch.long), slide_ids


class SlideRetrieval():
    def __init__(self, dataset):
        self.features = dataset.features
        self.labels = dataset.labels
        self.slide_ids = dataset.slide_ids

    def compute_similarity(self):
        """计算所有样本之间的余弦相似度"""
        feature_matrix = self.features.numpy()
        similarity_matrix = cosine_similarity(feature_matrix)
        return similarity_matrix

    def retrieve_top_k(self, similarity_matrix, k=5):
        """获取每个样本的 Top-K 结果"""
        top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, 1:k+1]  # 排除自己
        return top_k_indices

    def evaluate_retrieval(self, top_k_indices, k_list=[1, 3, 5]):
        """计算 Top-K accuracy 和 MV@K 指标"""
        top_k_acc = {k: 0 for k in k_list}
        mv_k_acc = {k: 0 for k in k_list}
        
        num_samples = len(self.labels)
        
        for i in range(num_samples):
            true_label = self.labels[i].item()
            retrieved_labels = [self.labels[idx].item() for idx in top_k_indices[i]]
            
            # 计算Top-K准确率
            for k in k_list:
                if true_label in retrieved_labels[:k]:
                    top_k_acc[k] += 1

                # 计算MV@K (majority voting)
                label_counts = Counter(retrieved_labels[:k])
                majority_label = label_counts.most_common(1)[0][0]
                if majority_label == true_label:
                    mv_k_acc[k] += 1
        
        # 计算最终准确率
        top_k_acc = {k: round(top_k_acc[k] / num_samples, 4) for k in k_list}
        mv_k_acc = {k: round(mv_k_acc[k] / num_samples, 4) for k in k_list}

        return top_k_acc, mv_k_acc


# === 运行检索流程 ===
feature_dirs = "/data4/embedding/TCGA-BRCA/UNI"
label_csv = "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv"

dataset = H5Dataset(feature_dirs, label_csv)
retrieval = SlideRetrieval(dataset)

# 计算余弦相似度
similarity_matrix = retrieval.compute_similarity()

# 获取Top-K检索结果
top_k_indices = retrieval.retrieve_top_k(similarity_matrix, k=5)

# 计算评估指标
top_k_acc, mv_k_acc = retrieval.evaluate_retrieval(top_k_indices, k_list=[1, 3, 5])

# 输出结果
print("=== Slide Retrieval Evaluation Results ===")
print(f"Top-1 Accuracy: {top_k_acc[1]}")
print(f"Top-3 Accuracy: {top_k_acc[3]}")
print(f"MV@3 Accuracy: {mv_k_acc[3]}")
print(f"Top-5 Accuracy: {top_k_acc[5]}")
print(f"MV@5 Accuracy: {mv_k_acc[5]}")
