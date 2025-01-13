import subprocess
import os

def run_task(config):
    gpu_id, task_cfg, dataset_csv, root_path, input_dim, pretrain_model, pretrain_model_type = config
    log_save_dir = "log"
    # Create a unique log file name
    log_file = os.path.join(log_save_dir, f"{os.path.basename(task_cfg)}_{os.path.basename(dataset_csv)}_{pretrain_model}.log")

    # If the log file exists, skip the task
    if os.path.exists(log_file):
        print(f"Skipping task: {log_file} already exists")
        return None  # Task skipped

    # Build the command
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
    print(f"Launching task: {command_str}")

    # Open the log file and launch the process
    log = open(log_file, "w")
    process = subprocess.Popen(command_str, shell=True, stdout=log, stderr=log)

    return process  # Return the subprocess object to track it

if __name__ == "__main__":
    # Dataset and embedding directory
    dataset = 'BRACS'
    embedding_dir = "/ruiyan/yuhao/tile_embed/"
    pretrain_models = ["CONCH", "UNI", "Virchow", "FMBC"]

    # Define paths and configurations
    root_paths = [os.path.join(embedding_dir, dataset, pretrain_model) for pretrain_model in pretrain_models]
    input_dims = [768, 1024, 1280, 768]
    length = len(pretrain_models)
    gpu_ids = [0] * length  # Using GPU ID 0 for all tasks
    task_cfg_paths = ["task_configs/bracs_coarse.yaml"] * length
    dataset_csvs = ["dataset_csv/subtype/BRACS_coarse.csv"] * length
    pretrain_model_types = ["patch_level", "patch_level", "patch_level", "slide_level"]

    # Create a list of all configurations
    configs = zip(gpu_ids, task_cfg_paths, dataset_csvs, root_paths, input_dims, pretrain_models, pretrain_model_types)

    # Start all tasks simultaneously
    processes = []
    for config in configs:
        process = run_task(config)
        if process:
            processes.append(process)  # Keep track of running processes

    # Wait for all processes to complete
    for process in processes:
        process.wait()

    print("All tasks completed.")
