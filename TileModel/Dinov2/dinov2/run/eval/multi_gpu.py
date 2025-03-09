import os
import time
import subprocess
import torch

def get_available_gpus(threshold=1000):
    """获取当前可用的 GPU 列表"""
    num_gpus = torch.cuda.device_count()
    available_gpus = []

    try:
        # 获取所有 GPU 的已使用显存
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip().split("\n")

        for i in range(num_gpus):
            free_memory = int(gpu_info[i].strip())  # 获取空闲显存
            print(f"GPU {i} has {free_memory} MB free memory")
            if free_memory > threshold:  # 设定阈值（空闲显存大于 1000MB）
                available_gpus.append(str(i))
    except Exception as e:
        print(f"Error checking GPU availability: {e}")

    print(f"Available GPUs: {available_gpus}")
    return available_gpus


def get_unprocessed_datasets(data_dir, processed_dir, feat_prefix_name):
    """获取未处理的数据集"""
    all_datasets = os.listdir(data_dir)
    processed_datasets = os.listdir(processed_dir) if os.path.exists(processed_dir) else []

    # 将 private_ 开头的数据集移动到队列后面
    all_datasets = [d for d in all_datasets if not d.startswith('private_')] + \
                   [d for d in all_datasets if d.startswith('private_')]

    # 移除不需要处理的数据集
    exclude_datasets = {"BreakHis", "HE-vs-MPM", "BACH"}
    all_datasets = [d for d in all_datasets if d not in exclude_datasets]

    unprocessed_datasets = []
    for d in all_datasets:
        save_path = os.path.join(processed_dir, d, feat_prefix_name)
        os.makedirs(save_path, exist_ok=True)

        # 判断是否有足够的未处理数据
        input_count = len(os.listdir(os.path.join(data_dir, d, 'output')))
        processed_count = len(os.listdir(save_path))
        if (input_count - processed_count) > 10:
            unprocessed_datasets.append(d)

    print(f"Unprocessed datasets: {unprocessed_datasets}")
    return unprocessed_datasets


def main():
    data_dir = '/data4/processed_data'
    save_dir = '/data4/embedding'
    script_path = 'feature_extrac.py'
    feat_prefix_name = 'FMBC'

    active_tasks = {}  # 记录当前正在运行的任务 {gpu_id: process}
    task_index = 0  # 用于 GPU 轮询分配

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
        unprocessed_datasets = get_unprocessed_datasets(data_dir, save_dir, feat_prefix_name)

        # 3. 任务调度：使用 Round-Robin 方式均匀分配任务
        while available_gpus and unprocessed_datasets:
            gpu = available_gpus[task_index % len(available_gpus)]  # 轮询选择 GPU
            task_index += 1  # 轮询索引增加

            if gpu in active_tasks:  # GPU 正在执行任务，跳过
                continue

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
