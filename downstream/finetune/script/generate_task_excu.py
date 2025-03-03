import os

# 预训练模型及其配置
pretrain_models = ['Gigapath_tile', 'CONCH', 'UNI', 'TITAN', 'Virchow', 'CHIEF_tile', 'Gigapath', 'CHIEF']
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

def get_tuning_methods(model_type):
    return ["ABMIL", "LR"] if model_type == "patch_level" else ["LR"]

# 任务配置
tasks = {
    # "BCNB_ALN": {
    #     "embedding_dir": "/data4/fm_embedding/embedding/BCNB",
    #     "csv_dir": "dataset_csv/subtype/",
    #     "dataset": "BCNB",
    #     "task_cfg": "task_configs/BCNB_ALN.yaml"
    # }, 
    # "AIDPATH_GRADE": {
    #     "embedding_dir": "/data4/fm_embedding/embedding/AIDPATH",
    #     "csv_dir": "dataset_csv/subtype/",
    #     "task_cfg": "task_configs/AIDPATH_GRADE.yaml"
    # }, 
    # "AIDPATH_IDC": {
    #     "embedding_dir": "/data4/fm_embedding/embedding/AIDPATH",
    #     "csv_dir": "dataset_csv/subtype/",
    #     "task_cfg": "task_configs/AIDPATH_IDC.yaml"
    # }, 
    # "IMPRESS_PR": {
    #     "embedding_dir": "/data4/fm_embedding/embedding/IMPRESS",
    #     "csv_dir": "dataset_csv/biomarker/",
    #     "dataset": "IMPRESS",
    #     "task_cfg": "task_configs/IMPRESS_PR.yaml"
    # },
    # 'SLNbreast_2subtype':{
    #     "embedding_dir": "/data4/fm_embedding/embedding/SLN-Breast",
    #     "csv_dir": "dataset_csv/subtype/",
    #     "dataset": "SLN-Breast",
    #     "task_cfg": "task_configs/SLNbreast_2subtype.yaml"
    # },
    'TCGA-BRCA-SUBTYPE':{
        "embedding_dir": "/data4/embedding/TCGA-BRCA",
        "csv_dir": "dataset_csv/subtype/",
        "dataset": "TCGA-BRCA",
        "task_cfg": "task_configs/TCGA-BRCA-SUBTYPE.yaml"
    }
}
root_dir = '/home/yuhaowang/project/FMBC/downstream/finetune'
# 生成命令并保存到文件
commands_file = "commands.txt"
#if exist(commands_file) delete the old
if os.path.exists(commands_file):
    os.remove(commands_file)
    
with open(commands_file, "w") as f:
    for task_name, config in tasks.items():
        embedding_dir = config["embedding_dir"]
        csv_dir = config["csv_dir"]
        task_cfg = config["task_cfg"]

        for pretrain_model in pretrain_models:
            pretrain_model_type = pretrain_model_types_dict[pretrain_model]
            tuning_methods = get_tuning_methods(pretrain_model_type)
            input_dim = pretrain_model_dim_dict[pretrain_model]
            root_path = os.path.join(embedding_dir, pretrain_model)
            dataset_csv = os.path.join(csv_dir, f"{task_name}.csv")
            
            for tuning_method in tuning_methods:
                output_prediction = root_dir+'/' + os.path.join('outputs', task_name, pretrain_model, tuning_method, 'prediction_results', 'val_predict.csv')
                #print(os.path.exists(output_prediction))
                if os.path.exists(output_prediction):
                    print(f"Skipping task: {output_prediction} already exists")
                    continue
                command = f"python main.py --task_cfg_path {task_cfg} --dataset_csv {dataset_csv} " \
                        f"--root_path {root_path} --input_dim {input_dim} --pretrain_model {pretrain_model} " \
                        f"--pretrain_model_type {pretrain_model_type} --tuning_method {tuning_method}\n"
                f.write(command)
                print(f"Generated command for {task_name} with {pretrain_model} {tuning_method}")

print(f"All commands have been saved to {commands_file}")
