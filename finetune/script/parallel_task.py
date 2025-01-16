import subprocess
import os

def run_task(config):
    gpu_id, task_cfg, dataset_csv, root_path, input_dim, pretrain_model, pretrain_model_type = config
    log_save_dir = "log"
    os.makedirs(log_save_dir, exist_ok=True)

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
    #change dir to ../
    os.chdir("../")
    # Open the log file and launch the process
    with open(log_file, "w") as log:
        process = subprocess.Popen(command_str, shell=True, stdout=log, stderr=log)
        process.wait()  # Wait for the process to finish

    print(f"Task completed: {command_str}")

if __name__ == "__main__":
    # Dataset and embedding directory
    datasets = ["Post-NAT-BRCA",]  # List of datasets
    task_name = "Post-NAT-3Type"
    embedding_dir = "/ruiyan/yuhao/embedding/"
    config_dir = "task_configs"
    csv_dir = "dataset_csv/subtype"
    pretrain_models = ["FMBC"]

    # Define configurations for each dataset
    input_dim = 768
    gpu_id = 0  # Set GPU ID for all tasks
    pretrain_model_type = "slide_level"

    for dataset in datasets:
        for pretrain_model in pretrain_models:
            root_path = os.path.join(embedding_dir, dataset)
            task_cfg_path = os.path.join(config_dir, f"{task_name}.yaml")
            dataset_csv = os.path.join(csv_dir, f"{task_name}.csv")

            config = (
                gpu_id, task_cfg_path, dataset_csv, root_path, input_dim, pretrain_model, pretrain_model_type
            )

            # Run the task and wait for its completion before starting the next
            run_task(config)

    print("All tasks completed.")
