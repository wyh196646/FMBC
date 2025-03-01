import os
import subprocess
import time

# 用户可配置的 GPU 列表及最大任务数
gpu_config = {
    0: 1,  # GPU 0 最多 4 个任务
    1: 1,  # GPU 1 最多 3 个任务
    2: 1,  # GPU 2 最多 2 个任务
    3: 1,  # GPU 3 最多 2 个任务
    4: 1,  # GPU 4 最多 2 个任务
    5: 1,  # GPU 5 最多 2 个任务
    6: 1,  # GPU 6 最多 2 个任务
    7: 1,  # GPU 7 最多 2 个任务
}
pretrain_models = ['FMBC']
pretrain_model_dim_dict = {

    "FMBC":1536,
}
pretrain_model_types_dict = {

    "FMBC": "slide_level"
}
def get_tuning_methods(pretrain_model):
    if pretrain_model =='FMBC':
        combinations = [
            f"LR_{lr}_{pool}"
            for lr in ["Frozen", "Same", "Different"]
            for pool in ["MeanPool", "CLSPool"]
        ]

    return combinations

# 任务配置
tasks = {
    "BCNB_ALN": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "dataset_csv/subtype/",
        "dataset": "BCNB",
        "task_cfg": "task_configs/BCNB_ALN.yaml"
    }, 
    "AIDPATH_GRADE": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "dataset_csv/subtype/",
        "task_cfg": "task_configs/AIDPATH_GRADE.yaml"
    }, 
    "AIDPATH_IDC": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "dataset_csv/subtype/",
        "task_cfg": "task_configs/AIDPATH_IDC.yaml"
    },
    "BACH_TUMOR": {
        "embedding_dir": "/data4/embedding/BACH",
        "csv_dir": "dataset_csv/subtype/",
        "dataset": "BACH",
        "task_cfg": "task_configs/BACH_TUMOR.yaml"
    }, 


    'SLNBREAST_SUBTYPE':{
        "embedding_dir": "/data4/embedding/SLN-Breast",
        "csv_dir": "dataset_csv/subtype/",
        "dataset": "SLN-Breast",
        "task_cfg": "task_configs/SLNBREAST_SUBTYPE.yaml"
    },
    'TCGA-BRCA-SUBTYPE':{
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype/",
        "dataset": "TCGA-BRCA",
        "task_cfg": "task_configs/TCGA-BRCA-SUBTYPE.yaml"
    },
    "IMPRESS_PR": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/biomarker/",
        "dataset": "IMPRESS",
        "task_cfg": "task_configs/IMPRESS_PR.yaml"
    },
}
def get_available_gpus():
    return list(gpu_config.keys())

def is_task_hanging(process):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], 
                            capture_output=True, text=True)
    output = result.stdout.strip()
    if str(process.pid) in output and "0 MiB" in output:
        return True
    return False

def run_task(task_name, command, gpu_id):
    print(f"Starting: {command} on GPU {gpu_id}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    process = subprocess.Popen(command, shell=True, env=env)
    running_tasks[gpu_id].append((process, command))

gpu_list = get_available_gpus()
running_tasks = {gpu: [] for gpu in gpu_list}

task_queue = []
for task_name, config in tasks.items():
    embedding_dir = config["embedding_dir"]
    csv_dir = config["csv_dir"]
    task_cfg = config["task_cfg"]
    
    for pretrain_model in pretrain_models:
        pretrain_model_type = pretrain_model_types_dict[pretrain_model]
        tuning_methods = get_tuning_methods(pretrain_model)
        input_dim = pretrain_model_dim_dict[pretrain_model]
        root_path = os.path.join(embedding_dir, pretrain_model)
        dataset_csv = os.path.join(csv_dir, f"{task_name}.csv")
        
        for tuning_method in tuning_methods:
            output_prediction = os.path.join('outputs', task_name, pretrain_model, tuning_method, 'prediction_results', 'val_predict.csv')
            
            if os.path.exists(output_prediction):
                print(f"Skipping task: {output_prediction} already exists")
                continue
            
            command = f"python main.py --task_cfg_path {task_cfg} --dataset_csv {dataset_csv} " \
                      f"--root_path {root_path} --input_dim {input_dim} --pretrain_model {pretrain_model} " \
                      f"--pretrain_model_type {pretrain_model_type} --tuning_method {tuning_method}"
            task_queue.append((task_name, command))

while task_queue or any(len(v) > 0 for v in running_tasks.values()):
    for gpu in gpu_list:
        running_tasks[gpu] = [(p, cmd) for p, cmd in running_tasks[gpu] if p.poll() is None]
        if len(running_tasks[gpu]) < gpu_config[gpu] and task_queue:
            task_name, cmd = task_queue.pop(0)
            run_task(task_name, cmd, gpu)
    
    time.sleep(10)

print("All tasks completed.")


