


import os
import subprocess

data_list = [
    # "AIDPATH", "BRACS", "CAMELYON17", "GEO-GSE243280", "IMPRESS",  "TCGA-BRCA",
    # "ACROBAT", "BCNB", "CAMELYON16", "CPTAC-BREAST-all", "HE-vs-MPM", "Multi-omic",  "private_chunk_test", "TIGER",
    # "BACH", "BreakHis", "CMB-BRCA", "GTEX_Breast", "HyReCo", "Post-NAT-BRCA",  "SLN-Breast", "TUPAC",
    "HIST2ST", "DORID", "TNBC"
]

print(len(data_list))
# # 配置参数
CONFIG = {
    "remote_host": "yuhaowang@172.16.120.21",  # 远程服务器地址
    "remote_password": "data@YHWang",
    "remote_base_path": "/mnt/data/ruiyan/processed_data/",  # 本地基础路径
    "folders_to_sync": data_list,
    "local_base_path": "/data2",  # 本地基础路径

}

def sync_folder(remote_host, remote_password, remote_base_path, local_base_path, folders_to_sync):
    """同步指定的子文件夹"""
    for folder in folders_to_sync:
        remote_path = os.path.join(remote_base_path, folder)
        local_path = os.path.join(local_base_path, folder)
        
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        
        command = [
            'sshpass', '-p', remote_password, 'rsync', '-avz', '--progress', '--delete',
            f'{remote_host}:{remote_path}/', local_path
        ]
        
        print(f"同步 {folder} ...")
        subprocess.run(command)

def main():
    remote_host = CONFIG['remote_host']  # 远程服务器地址
    remote_password = CONFIG['remote_password']  # 远程服务器密码
    remote_base_path = CONFIG['remote_base_path']  # 远程基础路径
    local_base_path = CONFIG['local_base_path']  # 本地基础路径
    folders_to_sync = CONFIG['folders_to_sync']  # 需要同步的子文件夹列表
    
    sync_folder(remote_host, remote_password, remote_base_path, local_base_path, folders_to_sync)
    print("同步完成！")

if __name__ == "__main__":
    main()
