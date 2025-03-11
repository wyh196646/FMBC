import os
import subprocess
import time

# 用户可配置的 GPU 列表及最大任务数
gpu_config = {
    0: 6,  
    1: 6, 
    2: 6, 
    3: 6,  
    4: 6, 
    5: 6,  
    6: 6,  
    7: 6,  
}
pretrain_models = ['Gigapath_tile', 'CONCH', 'UNI', 'TITAN', 'Virchow', 'CHIEF_tile', 'Gigapath', 'CHIEF','FMBC']
pretrain_model_dim_dict = {
    "UNI": 1024,
    "CONCH": 768,
    "CHIEF_tile": 768,
    "TITAN": 768,
    "Virchow": 2560,
    "Gigapath_tile": 1536,
    "Gigapath": 768,
    "CHIEF": 768,
    'FMBC':768,
}
pretrain_model_types_dict = {
    "UNI": "patch_level",
    "CONCH": "patch_level",
    "CHIEF_tile": "patch_level",
    "TITAN": "slide_level",
    "Virchow": "patch_level",
    "Gigapath_tile": "patch_level",
    "Gigapath": "slide_level",
    "CHIEF": "slide_level",
    'FMBC': 'patch_level',
}
# 根据模型类型返回调参方法
def get_tuning_methods(pretrain_model,model_type):
    if pretrain_model =='FMBC':
        return ["LR_Same_Patch"]
    # 如果模型类型为patch_level，则返回["LR"]，否则返回["LR"]
    return ["LR"] if model_type == "patch_level" else ["LR"]

learning_rates = [0.1,0.01,0.001, 0.0001]
tasks = {
    "TCGA-BRCA-SUBTYPE": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml"
    },
    "CAMELYON16_TEST_CANCER": {
        "embedding_dir": "/data4/embedding/CAMELYON16",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CAMELYON16_TEST_CANCER.yaml"
    },
    "DORID_2": {
        "embedding_dir": "/data4/embedding/DORID",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/DORID_2.yaml"
    },
    "TCGA-BRCA_T": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA_T.yaml"
    },
    "BRACS_COARSE": {
        "embedding_dir": "/data4/embedding/BRACS_COARSE",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/BRACS_COARSE.yaml"
    },
    "AHSL-NON-IDC-GRADE": {
        "embedding_dir": "/data4/embedding/AHSL",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AHSL-NON-IDC-GRADE.yaml"
    },
    "CAMELYON16_TEST_IDC": {
        "embedding_dir": "/data4/embedding/CAMELYON16",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CAMELYON16_TEST_IDC.yaml"
    },
    "BCNB_TUMOR": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/BCNB_TUMOR.yaml"
    },
    "BCNB_ALN": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/BCNB_ALN.yaml"
    },
    "AIDPATH_IDC": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AIDPATH_IDC.yaml"
    },
    "CAMELYON17_STAGE_4SUBTYPING": {
        "embedding_dir": "/data4/embedding/CAMELYON17",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CAMELYON17_STAGE_4SUBTYPING.yaml"
    },
    "IMPRESS_RESIDUAL-TUMOR": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/IMPRESS_RESIDUAL-TUMOR.yaml"
    },
    "AHSL-GRADE-3": {
        "embedding_dir": "/data4/embedding/AHSL",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AHSL-GRADE-3.yaml"
    },
    "BRACS_FINE": {
        "embedding_dir": "/data4/embedding/BRACS",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/BRACS_FINE.yaml"
    },
    "AHSL-GRADE-1": {
        "embedding_dir": "/data4/embedding/AHSL",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AHSL-GRADE-1.yaml"
    },
    "BACH_TUMOR": {
        "embedding_dir": "/data4/embedding/BACH",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/BACH_TUMOR.yaml"
    },
    "SLNBREAST_SUBTYPE": {
        "embedding_dir": "/data4/embedding/SLNBREAST",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/SLNBREAST_SUBTYPE.yaml"
    },
    "TCGA-BRCA_STAGE": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA_STAGE.yaml"
    },
    "POST-NAT-3TYPE": {
        "embedding_dir": "/data4/embedding/POST-NAT",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/POST-NAT-3TYPE.yaml"
    },
    "TCGA-BRCA_M": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA_M.yaml"
    },
    "DORID_6": {
        "embedding_dir": "/data4/embedding/DORID",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/DORID_6.yaml"
    },
    "CPTAC_IDC": {
        "embedding_dir": "/data4/embedding/CPTAC",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CPTAC_IDC.yaml"
    },
    "TUPAC_TUMOR_SCORE": {
        "embedding_dir": "/data4/embedding/TUPAC",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TUPAC_TUMOR_SCORE.yaml"
    },
    "AIDPATH_GRADE": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/AIDPATH_GRADE.yaml"
    },
    "TCGA-BRCA_N": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/TCGA-BRCA_N.yaml"
    },
    "POST-NAT-HERIHC": {
        "embedding_dir": "/data4/embedding/POST-NAT",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/POST-NAT-HERIHC.yaml"
    },
    "CAMELYON17_STAGE": {
        "embedding_dir": "/data4/embedding/CAMELYON17",
        "csv_dir": "dataset_csv/subtype",
        "task_cfg": "task_configs/subtype/CAMELYON17_STAGE.yaml"
    },
    "AIDPATH_GRADE": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "dataset_csv/subtype/",
        "task_cfg": "task_configs/subtype/AIDPATH_GRADE.yaml"
    }, 

    "TCGA-BRCA_MOLECULAR_SUBTYPING": {
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/TCGA-BRCA_MOLECULAR_SUBTYPING.yaml"
    },
    "IMPRESS_PCR": {
        "embedding_dir": "/data4/embedding/IMPRES",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_PCR.yaml"
    },
    "IMPRESS_PD-L1-TUMOR": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_PD-L1-TUMOR.yaml"
    },
    "IMPRESS_CD8-PERITUMORAL": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_CD8-PERITUMORAL.yaml"
    },
    "IMPRESS_PD-L1-STROMA": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_PD-L1-STROMA.yaml"
    },
    "IMPRESS_PR": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_PR.yaml"
    },
    "IMPRESS_CD163-PERITUMORAL": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_CD163-PERITUMORAL.yaml"
    },
    "BCNB_HER2": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/BCNB_HER2.yaml"
    },
    "AIDPATH_KI67PRED": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/AIDPATH_KI67PRED.yaml"
    },
    "POST-NAT-PR": {
        "embedding_dir": "/data4/embedding/POST-NAT",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/POST-NAT-PR.yaml"
    },
    "BCNB_ER": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/BCNB_ER.yaml"
    },
    "AIDPATH_CERB2": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/AIDPATH_CERB2.yaml"
    },
    "IMPRESS_CD163-INTRATUMORAL": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_CD163-INTRATUMORAL.yaml"
    },
    "BCNB_PR": {
        "embedding_dir": "/data4/embedding/BCNB",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/BCNB_PR.yaml"
    },
    "POST-NAT-ER": {
        "embedding_dir": "/data4/embedding/POST-NAT",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/POST-NAT-ER.yaml"
    },
    "IMPRESS_CD8-INTRATUMORAL": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_CD8-INTRATUMORAL.yaml"
    },
    "AIDPATH_RESTR": {
        "embedding_dir": "/data4/embedding/AIDPATH",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/AIDPATH_RESTR.yaml"
    },
    "POST-NAT-ANTIHER2": {
        "embedding_dir": "/data4/embedding/POST-NAT",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/POST-NAT-ANTIHER2.yaml"
    },
    "CPTAC_AJCC8SUBTYPE": {
        "embedding_dir": "/data4/embedding/CPTAC",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/CPTAC_AJCC8SUBTYPE.yaml"
    },
    "IMPRESS_ER": {
        "embedding_dir": "/data4/embedding/IMPRESS",
        "csv_dir": "/home/yuhaowang/project/FMBC/downstream/finetune/dataset_csv/biomarker",
        "task_cfg": "task_configs/biomarker/IMPRESS_ER.yaml"
    }
}

def get_available_gpus():
    return list(gpu_config.keys())

def is_task_hanging(process):
    # 运行nvidia-smi命令，查询计算应用的pid和使用的内存，并以csv格式输出，不显示表头
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], 
                            capture_output=True, text=True)
    # 获取命令输出结果
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
commands_file = "baseline.txt"
if os.path.exists(commands_file):
    os.remove(commands_file)
    
with open(commands_file, "w") as f:    
    for task_name, config in tasks.items():
        embedding_dir = config["embedding_dir"]
        csv_dir = config["csv_dir"]
        task_cfg = config["task_cfg"]
        
        for pretrain_model in pretrain_models:
            pretrain_model_type = pretrain_model_types_dict[pretrain_model]
            tuning_methods = get_tuning_methods(pretrain_model, pretrain_model_type)
            input_dim = pretrain_model_dim_dict[pretrain_model]
            root_path = os.path.join(embedding_dir, pretrain_model)
            dataset_csv = os.path.join(csv_dir, f"{task_name}.csv")


            for tuning_method in tuning_methods:
                for learning_rate in learning_rates:
                    output_prediction = os.path.join('outputs', task_name, pretrain_model, tuning_method, str(learning_rate), 'prediction_results', 'val_predict.csv')
                    
                    if os.path.exists(output_prediction):
                        print(f"Skipping task: {output_prediction} already exists")
                        continue
                    command = f"python main.py --task_cfg_path {task_cfg} --dataset_csv {dataset_csv} " \
                            f"--root_path {root_path} --input_dim {input_dim} --pretrain_model {pretrain_model} " \
                            f"--pretrain_model_type {pretrain_model_type} --tuning_method {tuning_method} --lr {learning_rate}"
                    task_queue.append((task_name, command))
                    save_command = command+'\n'
                    f.write(save_command)
    while task_queue or any(len(v) > 0 for v in running_tasks.values()):
        for gpu in gpu_list:
            running_tasks[gpu] = [(p, cmd) for p, cmd in running_tasks[gpu] if p.poll() is None]
            if len(running_tasks[gpu]) < gpu_config[gpu] and task_queue:
                task_name, cmd = task_queue.pop(0)
                run_task(task_name, cmd, gpu)
        
        time.sleep(10)

    print("All tasks completed.")
