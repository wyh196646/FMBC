from PIL import ImageFile, Image
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import shutil

# 设置以加载可能被截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 数据路径
data_path = '/yuhaowang/data/private_chunk_4'
image_paths = list(Path(data_path).rglob('*.png'))

# 定义验证函数
def validate_img(path):
    try:
        img = Image.open(path).convert(mode="RGB")  # 检查是否可以打开并转换为RGB
    except:
        os.remove(path)  # 如果失败，删除图像
        print('Removed:', path)

# 多进程处理
def main():
    num_processes = min(cpu_count(), 50)  # 使用的进程数，不超过CPU核心数
    print(f"Using {num_processes} processes...")
    with Pool(num_processes) as pool:
        list(pool.imap_unordered(validate_img, image_paths))  # 多进程处理文件

if __name__ == "__main__":
    main()

