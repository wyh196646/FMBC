import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir 
        self.ann_path = args.ann_path
        self.split_path = args.split_path

        self.max_seq_length = args.max_seq_length
        self.max_fea_length = args.max_fea_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        
        cases = self.clean_data(pd.read_csv(self.split_path).loc[:, self.split].dropna())

        
        self.examples = []
        root = self.ann_path

        count=0

        for dir in os.listdir(root):
            if not dir in cases.keys(): # check whther contained in the split
                continue
            else:
                img_name = cases[dir]
                
            image_path = os.path.join(self.image_dir,img_name)

            if not os.path.exists(image_path+'.pt'):
                continue
                
            file_name = os.path.join(root, dir, 'annotation')

            anno = json.loads(open(file_name, 'r').read())
            report_ids = tokenizer(anno)
            if len(report_ids) < self.max_seq_length:
                padding = [0] * (self.max_seq_length-len(report_ids))  
                report_ids.extend(padding)
            #report_ids = tokenizer(anno)[:self.max_seq_length]
            self.examples.append({'id':dir, 'image_path': image_path+'.pt','report': anno, 'split': self.split,'ids':report_ids, 'mask': [1]*len(report_ids)})

        
        print(f'The size of {self.split} dataset: {len(self.examples)}')


    def __len__(self):
        return len(self.examples)



class TcgaImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        
        image = torch.load(image_path)
        image = image[:self.max_fea_length]
        report_ids = example['ids']
        report_masks = example['mask']

        seq_length = len(report_ids)


        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


    def clean_data(self,data):
        cases = {}
        for idx in range(len(data)):
            case_name = data[idx]

            case_id = '-'.join(case_name.split('-')[:3])
            cases[case_id] = case_name
        return cases 
    
    def filter_df(self,df, filter_dict):
        # 如果filter_dict不为空
        if len(filter_dict) > 0:
            # 创建一个与df长度相同的布尔数组，初始值为True
            filter_mask = np.full(len(df), True, bool)
            # assert 'label' not in filter_dict.keys()
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def df_prep(self,data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data
    
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import TcgaImageDataset
import math

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'TCGA':
            self.dataset = TcgaImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

            
        if split == 'train':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            self.sampler = SequentialDistributedSampler(self.dataset,self.args.batch_size)


        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'sampler': self.sampler
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths= zip(*data)
        
        images = images[0].unsqueeze(0)

        return images_id, images, torch.LongTensor(reports_ids), torch.FloatTensor(reports_masks)


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples