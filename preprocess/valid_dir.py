import os
import numpy as np
import shutil

def valid_dir(dir_path):
    if len(os.listdir(dir_path)) <=3:
        shutil.rmtree(dir_path)
        print(f'{dir_path} is removed')

if __name__ == '__main__':
    dir_path = '/home/yuhaowang/data/processed_data/CAMELYON16/output'
    for dir in os.listdir(dir_path):
        valid_dir(os.path.join(dir_path, dir))