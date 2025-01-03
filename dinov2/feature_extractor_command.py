import subprocess

# 定义命令模板
base_command = (
    "nohup "
    "python dinov2/run/eval/feature_extractor.py "
    "--config-file dinov2/configs/train/patch.yaml "
    "--pretrained-weights /ruiyan/yuhao/project/output/eval/training_499999/teacher_checkpoint.pth "
    "--output-dir ./ "
    "--train-dataset TileDataset:split=TRAIN:root=/ruiyan/yuhao/data "
    "--dump_path {} "
    "--dataset_list {} "
    ">fea_ext.log 2>&1"
)

# 数据集列表
dataset_list = [
    'TCGA-BRCA',
    "ACROBAT",
    "BCNB", 
    "CAMELYON16", 
    # "CPTAC-BREAST-all", 
    # "Multi-omic", 
    # "SLN-Breast",
    # "TIGER",
    # "BACH",
    # "BRACS",
    # "CMB-BRCA",
    # "IMPRESS",
    # "Post-NAT-BRCA",
]


dump_path_base = "/ruiyan/yuhao/embedding/"
for dataset in dataset_list:
    # 替换 dump_path 和 dataset_list 的值
    dump_path = f"{dump_path_base}{dataset}"
    command = base_command.format(dump_path, dataset)
    print(f"正在执行命令: {command}")
    try:
        # 使用 subprocess.Popen 执行命令
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 实时打印标准输出和标准错误
        for line in process.stdout:
            print(line, end="")  # 标准输出内容
        for line in process.stderr:
            print(line, end="")  # 标准错误内容

        # 等待进程完成并检查退出状态
        process.wait()
        if process.returncode != 0:
            print(f"命令执行失败，退出代码: {process.returncode}")
            break

    except Exception as e:
        print(f"执行命令时出错: {e}")
        break

print("所有任务执行完毕！")
