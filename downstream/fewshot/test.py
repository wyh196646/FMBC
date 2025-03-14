import os
import h5py
import torch
import pandas as pd
import numpy as np
import torch
import numpy as np
import logging
from sklearn.datasets import make_classification
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot


class H5Dataset():
    def __init__(self, feature_dirs, label_csv):
        self.feature_dirs = feature_dirs
        self.features, self.labels = self.load_data(label_csv)
        #split the data into train and test sets, 0.8 for train and 0.2 for test
        self.train_features, self.train_labels, self.test_features, self.test_labels = self.split_data(self.features, self.labels, 0.8)
    def split_data(self, features, labels, train_ratio):
        '''Split the data into train and test sets'''
        train_size = int(len(features) * train_ratio)
        train_features = features[:train_size]
        train_labels = labels[:train_size]
        test_features = features[train_size:]
        test_labels = labels[train_size:]
        return train_features, train_labels, test_features, test_labels
    
    def read_assets_from_h5(self, h5_path):
        '''Read the assets from the h5 file'''
        assets = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                data = f[key][:]
                if len(data.shape) == 2:  # If shape is (N, 768), apply mean
                    data = np.mean(data, axis=0, keepdims=True)  # Shape becomes (1, 768)
                assets[key] = data
        return assets
    
    def load_data(self, label_csv):
        label_df = pd.read_csv(label_csv)
        labels = label_df['label'].values  # Extract label column
        
        all_features = []
        for folder in self.feature_dirs:
            for file in os.listdir(folder):
                if file.endswith(".h5"):
                    h5_path = os.path.join(folder, file)
                    assets = self.read_assets_from_h5(h5_path)
                    
                    # Assuming each h5 file has only one key
                    for key, feature in assets.items():
                        feature = feature.flatten()  # Ensure shape is (768,)
                        all_features.append(feature)
        
        all_features = np.array(all_features)  # Shape (N, 768
        
        return torch.tensor(all_features, dtype=torch.float32), torch.tensor(labels[:all_features.shape[0]], dtype=torch.long)

# Example usage
feature_dirs = ["model1", "model2", "model3"]  # Update with your actual paths

feature_dirs ='/data4/embedding/TCGA-BRCA/UNI'
label = '/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv'
dataset = H5Dataset(feature_dirs, label)

train_feats = dataset.train_features
train_labels = dataset.train_labels
test_feats = dataset.test_features
test_labels = dataset.test_labels

results_df, results_agg = eval_fewshot(train_feats, train_labels, test_feats, test_labels,
n_iter=10, n_way=5, n_shot=5, n_query=10,center_feats=True, normalize_feats=True, average_feats=True)

print("Few-shot Evaluation Results:")
print(results_df.head())
print("Aggregated Results:", results_agg)

