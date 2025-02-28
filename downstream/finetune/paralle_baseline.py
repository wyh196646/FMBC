import os
import subprocess
import time
import threading
import psutil

pretrain_models = ['Gigapath_tile','CONCH', 'UNI', 'TITAN','Virchow','CHIEF_tile','Gigapath','CHIEF']

pretrain_model_dim_dict = {
    "UNI": 1024,
    "CONCH": 768,
    "CHIEF_tile": 768,
    "TITAN": 768,
    "Virchow": 1280,
    "Gigapath_tile": 1536,
    "Gigapath": 768,
    "CHIEF": 768
}
pretrain_model_types_dict = {
    "UNI": "patch_level",
    "CONCH": "patch_level",
    "CHIEF_tile": "patch_level",
    "TITAN": "slide_level",
    "Virchow": "patch_level",
    "Gigapath_tile": "patch_level",
    "Gigapath": "slide_level",
    "CHIEF": "slide_level"
}
available_gpus = [0, 1, 2, 3, 4, 5 ,6, 7]
tasks = {
    "BCNB_ALN": {
        "embedding_dir": "/data4/fm_embedding/embedding/BCNB",
        "csv_dir": "dataset_csv/subtype/",
        "dataset": "BCNB",
        "task_cfg": "task_configs/BCNB_ALN.yaml"
    },
    "IMPRESS_HER2_2subtype": {
        "embedding_dir": "/data4/fm_embedding/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker/",
        "dataset": "IMPRESS",
        "task_cfg": "task_configs/IMPRESS_HER2_2subtype.yaml"
    },
    "IMPRESS_TNBC_2subtype": {
        "embedding_dir": "/data4/fm_embedding/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker/",
        "dataset": "IMPRESS",
        "task_cfg": "task_configs/IMPRESS_TNBC_2subtype.yaml"
    },
    'SLNbreast_2subtype':{
        "embedding_dir": "/data4/fm_embedding/embedding/SLN-Breast",
        "csv_dir": "dataset_csv/subtype/",
        "dataset": "SLN-Breast",
        "task_cfg": "task_configs/SLNbreast_2subtype.yaml"
    },
    'TCGA-BRCA-Subtype':{
        "embedding_dir": "/data4/fm_embedding/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype/",
        "dataset": "SLN-Breast",
        "task_cfg": "task_configs/TCGA-BRCA-Subtype.yaml"
    },
    
    "TCGA-BRCA_molecular_subtyping": {
        "embedding_dir": "/data4/fm_embedding/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/biomarker/",
        "dataset": "TCGA-BRCA",
        "task_cfg": "task_configs/TCGA-BRCA_molecular_subtyping.yaml"
    },
    "BRACS_COARSE":{
        "embedding_dir": "/data4/fm_embedding/embedding/BRACS",
        "csv_dir": "dataset_csv/subtype/",
        "dataset": "BRACS",
        "task_cfg": "task_configs/BRACS_COARSE.yaml"
        
    },
    "BRACS_FINE":{
        "embedding_dir": "/data4/fm_embedding/embedding/BRACS",
        "csv_dir": "dataset_csv/subtype/",
        "dataset": "BRACS",
        "task_cfg": "task_configs/BRACS_FINE.yaml"
        
    },
    
}

def get_tuning_methods(model_type):
    return ["ABMIL", "LR"] if model_type == "patch_level" else ["LR"]

def get_least_used_gpu(available_gpus):
    cmd = "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print("Error while checking GPU utilization")
        return None
    
    gpu_utilization = []
    lines = result.stdout.decode('utf-8').splitlines()
    
    for line in lines:
        gpu_data = line.split(', ')
        gpu_index = int(gpu_data[0])
        gpu_util = int(gpu_data[1])
        
        if gpu_index in available_gpus:
            gpu_utilization.append((gpu_index, gpu_util))
    
    if not gpu_utilization:
        return None
    
    return min(gpu_utilization, key=lambda x: x[1])[0]

def run_task(config, available_gpus):
    gpu_id, task_cfg, dataset_csv, root_path, input_dim, pretrain_model, pretrain_model_type, tuning_method = config
    log_save_dir = "./log"
    os.makedirs(log_save_dir, exist_ok=True)
    task_name = os.path.basename(task_cfg).split('.')[0]
    log_file = os.path.join(log_save_dir, f"{task_name}_{pretrain_model}_{tuning_method}.log")
    output_prediction = os.path.join('outputs', task_name, pretrain_model, tuning_method, 'prediction_results', 'val_predict.csv')

    if os.path.exists(output_prediction):
        print(f"Skipping task: {output_prediction} already exists")
        return None

    if gpu_id is None:
        gpu_id = get_least_used_gpu(available_gpus)
        if gpu_id is None:
            print("No available GPUs for the task.")
            return None

    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py --task_cfg_path {task_cfg} --dataset_csv {dataset_csv} --root_path {root_path} --input_dim {input_dim} --pretrain_model {pretrain_model} --pretrain_model_type {pretrain_model_type} --tuning_method {tuning_method}"
    
    with open(log_file, 'w') as f:
        f.write(f"GPU: {gpu_id}\n")
        f.write(f"Task Config: {task_cfg}\n")
        f.write(f"Dataset CSV: {dataset_csv}\n")
        f.write(f"Root Path: {root_path}\n")
        f.write(f"Input Dim: {input_dim}\n")
        f.write(f"Experiment Command: {command}\n")
    print(f"Launching task on GPU {gpu_id}: {command}")

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with open(log_file, "a") as log:
        for line in process.stdout:
            decoded_line = line.decode('utf-8')
            print(decoded_line, end='')  
            log.write(decoded_line)
        for line in process.stderr:
            decoded_line = line.decode('utf-8')
            print(decoded_line, end='')  
            log.write(decoded_line)
    
    return process, log_file

def manage_processes(configs, available_gpus, max_concurrent_tasks):
    all_processes = []
    threads = []

    def start_task(config):
        process, log_file = run_task(config, available_gpus)
        if process:
            all_processes.append((process, log_file))
    
    for config in configs:
        while len(all_processes) >= max_concurrent_tasks:
            for process, log_file in all_processes:
                if process.poll() is not None:
                    all_processes.remove((process, log_file))
                    break
            time.sleep(1)
        
        thread = threading.Thread(target=start_task, args=(config,))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    for process, log_file in all_processes:
        process.wait()
        if process.returncode != 0 and os.path.exists(log_file):
            print(f"Task failed. Deleting log file: {log_file}")
            raise Exception(f"Task failed. Deleting log file: {log_file}")

max_concurrent_tasks = 16

for task_name, config in tasks.items():
    embedding_dir = config["embedding_dir"]
    csv_dir = config["csv_dir"]
    task_cfg = config["task_cfg"]

    configs = []
    for pretrain_model in pretrain_models:
        pretrain_model_type = pretrain_model_types_dict[pretrain_model]
        tuning_methods = get_tuning_methods(pretrain_model_type)
        input_dim = pretrain_model_dim_dict[pretrain_model]
        root_path = os.path.join(embedding_dir, pretrain_model)
        dataset_csv = os.path.join(csv_dir, f"{task_name}.csv")
        
        for tuning_method in tuning_methods:
            configs.append((None, task_cfg, dataset_csv, root_path, input_dim, pretrain_model, pretrain_model_type, tuning_method))
    
    print(f"Starting tasks for {task_name}...")
    manage_processes(configs, available_gpus, max_concurrent_tasks)
    print(f"All tasks for {task_name} completed.")

print("All tasks for all task types completed.")
