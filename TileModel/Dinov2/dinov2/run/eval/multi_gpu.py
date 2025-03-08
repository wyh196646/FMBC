import os
import time
import subprocess
import torch

def get_available_gpus():
    """获取当前可用的 GPU 列表"""
    num_gpus = torch.cuda.device_count()
    available_gpus = []
    
    for i in range(num_gpus):
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
            ).decode("utf-8").strip().split("\n")
            used_memory = int(gpu_info[i])
            if used_memory < 1000:  # 阈值可调，表示空闲 GPU
                available_gpus.append(str(i))
        except Exception as e:
            print(f"Error checking GPU {i}: {e}")
    
    return available_gpus


def get_unprocessed_datasets(data_dir, processed_dir,feat_prefix_name):
    """获取未处理的数据集"""
    all_datasets = os.listdir(data_dir)
    #all_datasets = ['BRACS']
    processed_datasets = os.listdir(processed_dir) if os.path.exists(processed_dir) else []
    #reorder the all_datasets, the private_* to the end
    all_datasets = [d for d in all_datasets if not d.startswith('private_')] + [d for d in all_datasets if d.startswith('private_')]

    unprocessed_dataset = []
    for d in all_datasets:
        if not os.path.exists(os.path.join(processed_dir, d, feat_prefix_name)):
            os.makedirs(os.path.join(processed_dir, d, feat_prefix_name))
        if len(os.listdir(os.path.join(data_dir, d, 'output'))) - len(os.listdir(os.path.join(processed_dir, d, feat_prefix_name)))>10:
            unprocessed_dataset.append(d)
    return unprocessed_dataset

def main():
    data_dir = '/data4/processed_data'
    save_dir = '/data4/embedding'
    script_path = 'feature_extrac.py'
    feat_prefix_name = 'FMBC'
    
    active_tasks = {}  # 记录当前正在运行的任务 {gpu_id: process}

    while True:
        # 1. 检查正在运行的任务，移除已完成的任务
        finished_gpus = []
        for gpu, process in active_tasks.items():
            if process.poll() is not None:  # 进程已结束
                print(f"Task on GPU {gpu} finished")
                finished_gpus.append(gpu)

        # 移除已完成的任务
        for gpu in finished_gpus:
            del active_tasks[gpu]

        # 2. 获取可用 GPU 和未处理的数据集
        available_gpus = get_available_gpus()
        unprocessed_datasets = get_unprocessed_datasets(data_dir, save_dir,feat_prefix_name)

        # 3. 任务调度
        for gpu in available_gpus:
            if gpu in active_tasks:  # GPU 上已经有任务在运行
                continue
            if not unprocessed_datasets:  # 没有未处理数据
                break

            dataset = unprocessed_datasets.pop(0)  # 取出一个未处理的数据集
            print(f"Starting processing {dataset} on GPU {gpu}")

            # 启动新任务
            process = subprocess.Popen([
                "python", script_path, 
                "--dataset_name", dataset, 
                "--gpu", gpu, 
                "--img_dir", data_dir, 
                "--save_dir", save_dir,
                "--prefix_name", feat_prefix_name
            ])

            # 记录该 GPU 的任务
            active_tasks[gpu] = process

        if not active_tasks:  # 如果当前没有任务，则等待
            print("No available GPUs or datasets. Waiting...")
            time.sleep(30)
        else:
            time.sleep(10)  # 避免频繁检查

if __name__ == "__main__":
    main()
