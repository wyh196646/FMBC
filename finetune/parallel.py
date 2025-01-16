import subprocess
import os
import time

def run_task(config):
    gpu_id, task_cfg, dataset_csv, root_path, input_dim, pretrain_model, pretrain_model_type = config
    log_save_dir = "./log"
    os.makedirs(log_save_dir, exist_ok=True)
    task_name = os.path.basename(task_cfg).split('.')[0]
    # Create a unique log file name
    log_file = os.path.join(log_save_dir, f"{task_name}_{pretrain_model}.log")
    output_prediction = os.path.join('outputs',
                                     task_name,
                                     pretrain_model,
                                     'prediction_results',
                                     'val_predict.csv'
                                     )
    if os.path.exists(output_prediction):
        print(f"Skipping task: {output_prediction} already exists")
        return None

    # Make log file and initial content
    with open(log_file, 'w') as f:
        f.write(f"GPU: {gpu_id}\n")
        f.write(f"Task Config: {task_cfg}\n")
        f.write(f"Dataset CSV: {dataset_csv}\n")
        f.write(f"Root Path: {root_path}\n")
        f.write(f"Input Dim: {input_dim}\n")

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

    # Open the log file and launch the process, capturing output in real-time
    process = subprocess.Popen(command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print output to the terminal and log file in real-time
    with open(log_file, "a") as log:
        # Capture stdout
        for line in process.stdout:
            decoded_line = line.decode('utf-8')
            print(decoded_line, end='')  # Print to terminal
            log.write(decoded_line)      # Write to log file

        # Capture stderr
        for line in process.stderr:
            decoded_line = line.decode('utf-8')
            print(decoded_line, end='')  # Print to terminal
            log.write(decoded_line)      # Write to log file

    return process  # Return the subprocess object to track it


if __name__ == "__main__":
    # Define tasks with their configurations
    tasks = {
        "BCNB_ER": {
            "embedding_dir": "/ruiyan/yuhao/tile_embed/BCNB",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "BCNB",
            "task_cfg": "task_configs/BCNB_ER.yaml"
        }, 
        "BCNB_HER2": {
            "embedding_dir": "/ruiyan/yuhao/tile_embed/BCNB",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "BCNB",
            "task_cfg": "task_configs/BCNB_HER2.yaml"
        },
        "BCNB_PR": {
            "embedding_dir": "/ruiyan/yuhao/tile_embed/BCNB",
            "csv_dir": "dataset_csv/biomarker/",
            "dataset": "BCNB",
            "task_cfg": "task_configs/BCNB_PR.yaml"
        }
    }

    # Define pretrain models to process in each batch
    pretrain_models = ['Gigapath_tile', 'CONCH', 'UNI', 'CHIEF_tile']
    pretrain_model_dim_dict = {
        "UNI": 1024,
        "CONCH": 768,
        "CHIEF_tile": 1536,
        "TITAN": 768,
        "Virchow": 1280,
        "Gigapath_tile": 1536,
    }
    pretrain_model_types_dict = {
        "UNI": "patch_level",
        "CONCH": "patch_level",
        "CHIEF_tile": "patch_level",
        "TITAN": "patch_level",
        "Virchow": "patch_level",
        "Gigapath_tile": "patch_level"
    }

    # Maximum number of concurrent tasks
    max_concurrent_tasks = 16  # Adjust this to your hardware capacity

    # Track all processes across all task types
    all_processes = []
    running_processes = 0  # Counter for the number of currently running tasks

    for task_name, config in tasks.items():
        embedding_dir = config["embedding_dir"]
        csv_dir = config["csv_dir"]
        dataset = config["dataset"]
        input_dim = [pretrain_model_dim_dict[pretrain_model] for pretrain_model in pretrain_models]
        pretrain_model_types = [pretrain_model_types_dict[pretrain_model] for pretrain_model in pretrain_models]

        task_cfg = config["task_cfg"]

        root_paths = [os.path.join(embedding_dir, pretrain_model) for pretrain_model in pretrain_models]
        length = len(pretrain_models)
        gpu_ids = [0 for i in range(length)]  # Distribute tasks across GPUs
        dataset_csvs = [os.path.join(csv_dir, f"{task_name}.csv")] * length

        # Create a list of all configurations
        configs = zip(gpu_ids, [task_cfg] * length, dataset_csvs, root_paths, input_dim, pretrain_models, pretrain_model_types)

        # Start all tasks for the current task type
        for config in configs:
            # Wait for a running process to finish if the number of concurrent tasks reaches the limit
            if running_processes >= max_concurrent_tasks:
                # Wait for one of the processes to complete
                for process in all_processes:
                    if process.poll() is not None:  # Process has finished
                        running_processes -= 1  # Decrease the counter for running tasks
                        all_processes.remove(process)  # Remove the finished process
                        break
                time.sleep(1)  # Sleep for a short time before checking again

            # Run the new task
            process = run_task(config)
            if process:
                all_processes.append(process)  # Keep track of running processes
                running_processes += 1  # Increase the counter for running tasks

        print(f"All tasks for {task_name} started.")

    # Wait for all processes of all task types to complete
    for process in all_processes:
        process.wait()

    print("All tasks for all task types completed.")
