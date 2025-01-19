# import subprocess
# import os

# def run_task(config):
#     gpu_id, task_cfg, dataset_csv, root_path, input_dim, pretrain_model, pretrain_model_type = config
#     log_save_dir = "log"
#     os.makedirs(log_save_dir, exist_ok=True)

#     # Create a unique log file name
#     log_file = os.path.join(log_save_dir, f"{os.path.basename(task_cfg)}_{os.path.basename(dataset_csv)}_{pretrain_model}.log")

#     # If the log file exists, skip the task
#     if os.path.exists(log_file):
#         print(f"Skipping task: {log_file} already exists")
#         return None  # Task skipped

#     # Build the command
#     command = [
#         f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py",
#         f"--task_cfg_path {task_cfg}",
#         f"--dataset_csv {dataset_csv}",
#         f"--root_path {root_path}",
#         f"--input_dim {input_dim}",
#         f"--pretrain_model {pretrain_model}",
#         f"--pretrain_model_type {pretrain_model_type}"
#     ]

#     command_str = " ".join(command)
#     print(f"Launching task: {command_str}")
#     #change dir to ../
#     os.chdir("../")
#     # Open the log file and launch the process
#     with open(log_file, "w") as log:
#         process = subprocess.Popen(command_str, shell=True, stdout=log, stderr=log)
#         process.wait()  # Wait for the process to finish

#     print(f"Task completed: {command_str}")

# if __name__ == "__main__":
#     # Dataset and embedding directory
#     datasets = ["Post-NAT-BRCA",]  # List of datasets
#     task_name = "Post-NAT-3Type"
#     embedding_dir = "/ruiyan/yuhao/embedding/"
#     config_dir = "task_configs"
#     csv_dir = "dataset_csv/subtype"
#     pretrain_models = ["FMBC"]

#     # Define configurations for each dataset
#     input_dim = 768
#     gpu_id = 0  # Set GPU ID for all tasks
#     pretrain_model_type = "slide_level"

#     for dataset in datasets:
#         for pretrain_model in pretrain_models:
#             root_path = os.path.join(embedding_dir, dataset)
#             task_cfg_path = os.path.join(config_dir, f"{task_name}.yaml")
#             dataset_csv = os.path.join(csv_dir, f"{task_name}.csv")

#             config = (
#                 gpu_id, task_cfg_path, dataset_csv, root_path, input_dim, pretrain_model, pretrain_model_type
#             )

#             # Run the task and wait for its completion before starting the next
#             run_task(config)

#     print("All tasks completed.")



import subprocess
import os
import time
import threading

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
    #write the command
    command_str = " ".join(command)
    with open(log_file, 'w') as f:
        f.write(f"GPU: {gpu_id}\n")
        f.write(f"Task Config: {task_cfg}\n")
        f.write(f"Dataset CSV: {dataset_csv}\n")
        f.write(f"Root Path: {root_path}\n")
        f.write(f"Input Dim: {input_dim}\n")
        f.write(f"Experiment Command: {command_str}\n")
    print(f"Launching task: {command_str}")

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


def manage_processes(configs, max_concurrent_tasks):
    all_processes = []
    running_processes = 0

    def start_task(config):
        nonlocal running_processes
        process, log_file = run_task(config)
        if process:
            all_processes.append((process, log_file))
            running_processes += 1

    threads = []

    for config in configs:
        while running_processes >= max_concurrent_tasks:
            for process, log_file in all_processes:
                if process.poll() is not None:  # Process finished
                    running_processes -= 1
                    all_processes.remove((process, log_file))
                    break
            time.sleep(1)

        # Launch task in a separate thread to allow parallel execution
        thread = threading.Thread(target=start_task, args=(config,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Wait for all subprocesses to finish
    for process, log_file in all_processes:
        process.wait()
        # If the process failed (non-zero return code), delete the log file
        if process.returncode != 0 and os.path.exists(log_file):
            print(f"Task failed. Deleting log file: {log_file}")
            os.remove(log_file)


if __name__ == "__main__":
    tasks = {
        "BCNB_ERPRHER2": {
            "embedding_dir": "/yuhaowang/data/embedding/BCNB",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "BCNB",
            "task_cfg": "task_configs/BCNB_ERPRHER2.yaml"
        }, 
        "BCNB_ALN": {
            "embedding_dir": "/yuhaowang/data/embedding/BCNB",
            "csv_dir": "dataset_csv/subtype/",
            "dataset": "BCNB",
            "task_cfg": "task_configs/BCNB_ALN.yaml"
        },
        "IMPRESS_HER2_2subtype": {
            "embedding_dir": "/yuhaowang/data/embedding/IMPRESS",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "IMPRESS",
            "task_cfg": "task_configs/IMPRESS_HER2_2subtype.yaml"
        },
        "IMPRESS_TNBC_2subtype": {
            "embedding_dir": "/yuhaowang/data/embedding/IMPRESS",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "IMPRESS",
            "task_cfg": "task_configs/IMPRESS_TNBC_2subtype.yaml"
        },
        'SLNbreast_2subtype':{
            "embedding_dir": "/yuhaowang/data/embedding/SLN-Breast",
            "csv_dir": "dataset_csv/subtype/",
            "dataset": "SLN-Breast",
            "task_cfg": "task_configs/SLNbreast_2subtype.yaml"
        },
        'TCGA-BRCA-Subtype':{
            "embedding_dir": "/yuhaowang/data/embedding/TCGA-BRCA",
            "csv_dir": "dataset_csv/subtype/",
            "dataset": "SLN-Breast",
            "task_cfg": "task_configs/TCGA-BRCA-Subtype.yaml"
        },
        "TCGA-BRCA_molecular_subtyping": {
            "embedding_dir": "yuhaowang/data/embedding/TCGA-BRCA",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "TCGA-BRCA",
            "task_cfg": "task_configs/TCGA-BRCA_molecular_subtyping.yaml"
        },

        "BRACS_COARSE":{
            "embedding_dir": "yuhaowang/data/embedding/BRACS",
            "csv_dir": "dataset_csv/subtype/",
            "dataset": "BRACS",
            "task_cfg": "task_configs/BRACS_COARSE.yaml"
            
        },
        "BRACS_FINE":{
            "embedding_dir": "yuhaowang/data/embedding/BRACS",
            "csv_dir": "dataset_csv/subtype/",
            "dataset": "BRACS",
            "task_cfg": "task_configs/BRACS_FINE.yaml"
            
        },
        "TCGA-Genexp": {
            "embedding_dir": "yuhaowang/data/embedding/TCGA-BRCA",
            "csv_dir": "dataset_csv/expression_prediction/",
            "dataset": "TCGA-BRCA",
            "task_cfg": "task_configs/TCGA-BRCA-Gene-Exp.yaml"
        },
        
    
        
    }

    #pretrain_models = ['Gigapath_Tile','CONCH', 'UNI', 'TITAN','Virchow','CHIEF_Tile']
    pretrain_models =['FMBC']
    pretrain_model_dim_dict = {
        "UNI": 1024,
        "CONCH": 768,
        "CHIEF_Tile": 768,
        "TITAN": 768,
        "Virchow": 1280,
        "Gigapath_Tile": 1536,
        'FMBC':768
    }
    pretrain_model_types_dict = {
        "UNI": "patch_level",
        "CONCH": "patch_level",
        "CHIEF_Tile": "patch_level",
        "TITAN": "slide_level",
        "Virchow": "patch_level",
        "Gigapath_Tile": "patch_level",
        'FMBC': "slide_level"
    }

    max_concurrent_tasks = 16

    for task_name, config in tasks.items():
        embedding_dir = config["embedding_dir"]
        csv_dir = config["csv_dir"]
        task_cfg = config["task_cfg"]

        input_dim = [pretrain_model_dim_dict[pretrain_model] for pretrain_model in pretrain_models]
        pretrain_model_types = [pretrain_model_types_dict[pretrain_model] for pretrain_model in pretrain_models]
        root_paths = [embedding_dir for pretrain_model in pretrain_models]
        dataset_csvs = [os.path.join(csv_dir, f"{task_name}.csv")] * len(pretrain_models)
        gpu_ids = [0 for _ in range(len(pretrain_models))]  # Distribute tasks across GPUs

        configs = zip(gpu_ids, [task_cfg] * len(pretrain_models), dataset_csvs, root_paths, input_dim, pretrain_models, pretrain_model_types)
        
        print(f"Starting tasks for {task_name}...")
        manage_processes(configs, max_concurrent_tasks)
        print(f"All tasks for {task_name} completed.")
    
    print("All tasks for all task types completed.")
