import os
from multiprocessing import Pool, cpu_count
from PIL import Image

def is_image_corrupt(file_path):
    """检查图片是否损坏"""
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图片是否完整
        return False  # 图片未损坏
    except (IOError, SyntaxError):
        return True  # 图片损坏

def process_file(file_path):
    """检查并删除损坏的图片"""
    if is_image_corrupt(file_path):
        os.remove(file_path)  # 删除损坏图片
        print('delete',file_path)
        return f"Deleted corrupt image: {file_path}"
    
    return None

def find_all_images(directory):
    """获取目录下所有图片文件路径"""
    image_extensions = { ".png"}
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                all_files.append(os.path.join(root, file))
    return all_files

def delete_corrupt_images(directory):
    """并行检查并删除目录中的损坏图片"""
    all_images = find_all_images(directory)
    print(f"Found {len(all_images)} images to check.")

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, all_images)

    # 打印删除的文件
    for result in results:
        if result:
            print(result)

if __name__ == "__main__":
    dataset_directory = "/ruiyan/yuhao/data/"  # 替换为你的数据集路径
    delete_corrupt_images(dataset_directory)
