


import argparse
import subprocess
import os
import time
import threading
from queue import Queue

def parse_args():
    parser = argparse.ArgumentParser(description="Manage FMBC model fine-tuning tasks.")
    parser.add_argument('--gpu_ids', type=str, required=True, help="Comma-separated list of GPU IDs to use.")
    parser.add_argument('--max_concurrent_tasks', type=int, default=2, help="Maximum concurrent tasks across GPUs.")
    return parser.parse_args()

def run_task(config):
    gpu_id, task_cfg, dataset_csv, root_path, input_dim, pretrain_model, pretrain_model_type = config
    log_save_dir = "./log"
    os.makedirs(log_save_dir, exist_ok=True)
    task_name = os.path.basename(task_cfg).split('.')[0]
    log_file = os.path.join(log_save_dir, f"{task_name}_{pretrain_model}.log")
    output_prediction = os.path.join('outputs', task_name, pretrain_model, 'prediction_results', 'val_predict.csv')

    if os.path.exists(output_prediction):
        print(f"Skipping task: {output_prediction} already exists")
        return None

    command = [
        f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py",
        f"--task_cfg_path {task_cfg}",
        f"--dataset_csv {dataset_csv}",
        f"--root_path {root_path}",
        f"--input_dim {input_dim}",
        f"--pretrain_model {pretrain_model}",
        f"--pretrain_model_type {pretrain_model_type}"
    ]
    command_str = " ".join(command)

    with open(log_file, 'w') as f:
        f.write(f"GPU: {gpu_id}\n")
        f.write(f"Task Config: {task_cfg}\n")
        f.write(f"Experiment Command: {command_str}\n")
    
    print(f"Launching task on GPU {gpu_id}: {command_str}")
    process = subprocess.Popen(command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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

def manage_processes(configs, gpu_list, max_concurrent_tasks):
    queue = Queue()
    for config in configs:
        queue.put(config)
    
    active_processes = {}
    
    while not queue.empty() or active_processes:
        # Check running processes
        for gpu_id in list(active_processes.keys()):
            process, log_file = active_processes[gpu_id]
            if process.poll() is not None:  # Process finished
                print(f"Task on GPU {gpu_id} completed. Checking for next task...")
                active_processes.pop(gpu_id)

        # Assign new tasks to free GPUs
        for gpu_id in gpu_list:
            if gpu_id not in active_processes and not queue.empty():
                config = queue.get()
                new_process, log_file = run_task(config)
                if new_process:
                    active_processes[gpu_id] = (new_process, log_file)
                time.sleep(1)
        
        time.sleep(5)  # Wait before checking again

def main():
    args = parse_args()
    gpu_list = [int(g) for g in args.gpu_ids.split(',')]
    max_concurrent_tasks = args.max_concurrent_tasks
    
    tasks = {
        "BCNB_ERPRHER2": {
            "embedding_dir": "/ruiyan/yuhao/embedding/BCNB",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "BCNB",
            "task_cfg": "task_configs/BCNB_ERPRHER2.yaml"
        }, 
        "BCNB_ALN": {
            "embedding_dir": "/ruiyan/yuhao/embedding/BCNB",
            "csv_dir": "dataset_csv/subtype/",
            "dataset": "BCNB",
            "task_cfg": "task_configs/BCNB_ALN.yaml"
        },
        "IMPRESS_HER2_2subtype": {
            "embedding_dir": "/ruiyan/yuhao/embedding/IMPRESS",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "IMPRESS",
            "task_cfg": "task_configs/IMPRESS_HER2_2subtype.yaml"
        },
        "IMPRESS_TNBC_2subtype": {
            "embedding_dir": "/ruiyan/yuhao/embedding/IMPRESS",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "IMPRESS",
            "task_cfg": "task_configs/IMPRESS_TNBC_2subtype.yaml"
        },
        'SLNbreast_2subtype':{
            "embedding_dir": "/ruiyan/yuhao/embedding/SLN-Breast",
            "csv_dir": "dataset_csv/subtype/",
            "dataset": "SLN-Breast",
            "task_cfg": "task_configs/SLNbreast_2subtype.yaml"
        },

        'TCGA-BRCA-Subtype':{
            "embedding_dir": "/ruiyan/yuhao/embedding/TCGA-BRCA",
            "csv_dir": "dataset_csv/subtype/",
            "dataset": "SLN-Breast",
            "task_cfg": "task_configs/TCGA-BRCA-Subtype.yaml"
        },
        "TCGA-BRCA_molecular_subtyping": {
            "embedding_dir": "/ruiyan/yuhao/embedding/TCGA-BRCA",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "TCGA-BRCA",
            "task_cfg": "task_configs/TCGA-BRCA_molecular_subtyping.yaml"
        },

        "BRACS_COARSE":{
            "embedding_dir": "/ruiyan/yuhao/embedding/BRACS",
            "csv_dir": "dataset_csv/subtype/",
            "dataset": "BRACS",
            "task_cfg": "task_configs/BRACS_COARSE.yaml"
            
        },
        "BRACS_FINE":{
            "embedding_dir": "/ruiyan/yuhao/embedding/BRACS",
            "csv_dir": "dataset_csv/subtype/",
            "dataset": "BRACS",
            "task_cfg": "task_configs/BRACS_FINE.yaml"
            
        },
        "TCGA-Genexp": {
            "embedding_dir": "/ruiyan/yuhao/embedding/TCGA-BRCA",
            "csv_dir": "dataset_csv/expression_prediction/",
            "dataset": "TCGA-BRCA",
            "task_cfg": "task_configs/TCGA-BRCA-Gene-Exp.yaml"
        },
        
    }
    
    pretrain_model = 'FMBC'
    pretrain_model_dim = 1535
    pretrain_model_type = "slide_level"
    
    configs = []
    for task_name, config in tasks.items():
        embedding_dir = config["embedding_dir"]
        csv_dir = config["csv_dir"]
        task_cfg = config["task_cfg"]

        root_path = embedding_dir + '/' + pretrain_model
        dataset_csv = os.path.join(csv_dir, f"{task_name}.csv")
        configs.append((0, task_cfg, dataset_csv, root_path, pretrain_model_dim, pretrain_model, pretrain_model_type))
    
    print("Starting task management...")
    manage_processes(configs, gpu_list, max_concurrent_tasks)
    print("All tasks completed.")

if __name__ == "__main__":
    main()
