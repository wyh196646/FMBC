import os
import paramiko
import argparse
from scp import SCPClient
from filecmp import dircmp

def create_ssh_client(host, port, username, password):
    """创建 SSH 客户端"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)
    return ssh

def get_remote_file_list(ssh, remote_dir):

    sftp = ssh.open_sftp()
    file_list = []

    def list_files(directory):
        for item in sftp.listdir_attr(directory):
            item_path = os.path.join(directory, item.filename)
            if item.st_mode & 0o40000:  # 如果是目录
                list_files(item_path)
            else:
                file_list.append(item_path)

    list_files(remote_dir)
    sftp.close()
    return file_list

def compare_and_sync(ssh, local_dir, remote_dir):
    sftp = ssh.open_sftp()
    scp = SCPClient(ssh.get_transport())


    remote_files = get_remote_file_list(ssh, remote_dir)

    local_files = []
    for root, _, files in os.walk(local_dir):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), local_dir)
            local_files.append(relative_path)

    for remote_file in remote_files:
        relative_path = os.path.relpath(remote_file, remote_dir)
        local_file_path = os.path.join(local_dir, relative_path)
        try:
            remote_file_stat = sftp.stat(remote_file)
            if not os.path.exists(local_file_path) or os.path.getsize(local_file_path) != remote_file_stat.st_size:

                local_dir_path = os.path.dirname(local_file_path)
                os.makedirs(local_dir_path, exist_ok=True)

     
                print(f"同步文件: {remote_file} -> {local_file_path}")
                scp.get(remote_file, local_file_path)
        except Exception as e:
            print(f"跳过文件 {remote_file}: {e}")

    scp.close()
    sftp.close()

def main():
    #argparser
    argparser = argparse.ArgumentParser(description='Data Sync')
    argparser.add_argument('--host', type=str, default='172.16.120.24', help='Remote host')
    argparser.add_argument('--port', type=int, default=22, help='Remote port')
    argparser.add_argument('--username', type=str, default='yuhaowang', help='Remote username')
    argparser.add_argument('--password', type=str, default='4090@YHWang', help='Remote password')
    argparser.add_argument('--remote_dir', type=str, default='/data1/embedding', help='Remote directory')
    argparser.add_argument('--local_dir', type=str, default='/ruiyan/yuhao/tile_embed/embedding', help='Local directory')
    
    args = argparser.parse_args()

    host = args.host
    port = args.port
    username = args.username
    password = args.password
    remote_dir = args.remote_dir
    local_dir = args.local_dir

    print(f"开始同步数据: {remote_dir} -> {local_dir}")

    ssh = create_ssh_client(host, port, username, password)
    try:
        compare_and_sync(ssh, local_dir, remote_dir)
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
