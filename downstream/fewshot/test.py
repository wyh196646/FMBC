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
        self.features, self.labels = self.load_data(feature_dirs,label_csv)
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
    
    def read_assets_from_h5(self,h5_path: str):
        """Read the assets from the h5 file"""
        assets = {}
        attrs = {}

        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs
    
    def load_data(self,feature_dirs , label_csv):
        label_df = pd.read_csv(label_csv)
        #ay be some h5 file is missing, how to get the valid label
        for file in label_df['slide_id']:
            if file+'.h5' not in os.listdir(feature_dirs):
                label_df = label_df[label_df['slide_id'] != file]
                
        labels = label_df['label'].values  
        label_dict ={
            label:num for num,label in enumerate(np.unique(labels))
        }
        target_labels = np.array([label_dict[label] for label in labels])
        all_features = []
        slide_id_list =label_df['slide_id'].values
        #
        
        for file in slide_id_list:
            h5_path = os.path.join(feature_dirs, file+'.h5')
            assets,_ = self.read_assets_from_h5(h5_path)
            feature = assets['features']
   
            # for key, feature in assets.items():
            #     feature = feature.flatten()  # Ensure shape is (768,)
            #     all_features.append(feature)
            if feature.shape[0]>1:
                #mean_pooling
                feature = np.mean(feature, axis=0)
            else:
                feature = feature.flatten()
            all_features.append(feature)
    
        all_features = np.array(all_features)  # Shape (N, 768
        
        return torch.tensor(all_features, dtype=torch.float32), torch.tensor(target_labels, dtype=torch.long)

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
n_iter=10,
n_way = 2,  #
n_shot=6,
 n_query=15,center_feats=True, normalize_feats=True)

print("Few-shot Evaluation Results:")
print(results_df.head())
print("Aggregated Results:", results_agg)

# n_way（几分类）：每个任务中有多少个类别。
# n_shot（每类多少个样本）：每个类别中有多少个支持样本（support set）。
# n_query（每类多少个查询样本）：用于评估的测试样本数（query set）。
# n_iter（迭代次数）：总共执行多少次 few-shot 任务。

