import os
import argparse
import time
import logging
import numpy as np
import openslide
import random
from multiprocessing import Pool, Value, Lock
from tqdm import tqdm

from functools import partial


parser = argparse.ArgumentParser(description='Generate the patch of tumor')
parser.add_argument('--wsi_path', default='', metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')
parser.add_argument('--mask_path', default='', metavar='MASK_PATH', type=str,
                    help='Path to the tumor mask of the input WSI file')
parser.add_argument('--patch_path', default='', metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                    'default 768')
parser.add_argument('--level', default=20, type=int, help='patch level, '
                    'default 20')
parser.add_argument('--num_process', default=50, type=int,
                    help='number of mutli-process, default 5')

count = Value('i', 0)
lock = Lock()

def process(path,args=None):
    wsi_name = os.path.basename(path)[:-4]
    patch_dir = os.path.join(args.patch_path, wsi_name)
    if  os.path.exists(patch_dir):
        pass
    else:
        os.mkdir(patch_dir)

    mask_path = os.path.join(args.mask_path, wsi_name + '.npy')
    
    mask = np.load(mask_path)
    slide = openslide.OpenSlide(path)
    try:
        mpp = round( float( slide.properties[openslide.PROPERTY_NAME_MPP_X] ), 1 )#这是切片的扫描层级，就是每像素代表0.25um病理，最原始的扫描值
    except:
        mpp = 0.3
    if mpp == 0.1:
        max_level = 80
    elif mpp == 0.3 or mpp == 0.2:
        max_level = 40
    elif mpp == 0.5:
        max_level = 20
    else:
        raise ValueError('当前分辨率为{}，不存在该倍数设置'.format(str(mpp)))
    X_slide, Y_slide = slide.level_dimensions[0]
    rate = round(max_level // args.level)
    cur_level = round(rate / slide.level_downsamples[1] )
    patch_level = round(X_slide / mask.shape[0])
    patch_size = round(rate / slide.level_downsamples[cur_level]) * args.patch_size
    step = int(args.patch_size / patch_level)
    #print(path, slide.level_count, rate, X_slide, Y_slide)
    print('当前图像最高倍数为{}倍，level1图像倍数为{}倍，取图倍数为{}倍，图像尺寸为{}'.format(max_level, slide.level_downsamples[1], cur_level, patch_size))
    slide.close()
    X_mask, Y_mask = mask.shape
    ori_x, ori_y = list((map(lambda x: np.int32(x/rate/step), np.where(mask))))
    coords = list(set(zip(ori_x, ori_y)))
    random.shuffle(coords)
    print('当前图像共有{}个坐标点'.format(len(coords)))
    if wsi_processing_check(coords,patch_dir):
        print('{} has been processed'.format(wsi_name))
        return
    num = 0
    for idx in tqdm(range(len(coords))):
        x_mask, y_mask = coords[idx][0], coords[idx][1]
        x_center = int((x_mask + 0.5) * patch_level * step * rate)
        y_center = int((y_mask + 0.5) * patch_level * step * rate)
        x = int(int(x_center) - args.patch_size * rate / 2)
        y = int(int(y_center) - args.patch_size * rate / 2)
        img_path = os.path.join(patch_dir, os.path.basename(path)[:-4] + '_' + str(x) + '_' + str(y) + '.png')
        if os.path.exists(img_path):
            continue
        opt=(num, path, patch_dir, x, y, rate, args, patch_size, cur_level)
        num = num + 1
        i, wsi_path, patch_dir, x, y, rate, args, patch_size, cur_level = opt
        slide = openslide.OpenSlide(wsi_path)
        img = slide.read_region(
            (x, y), cur_level,
            (patch_size, patch_size)).convert('RGB')

        if patch_size != args.patch_size:
            img = img.resize((args.patch_size, args.patch_size))
        img_path = os.path.join(patch_dir, os.path.basename(wsi_path)[:-4] + '_' + str(x) + '_' + str(y) + '.png')
        img.save(img_path)


    global lock
    global count

    with lock:
        count.value += 1
        if (count.value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 count.value))


def run(args):
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(args.mask_path):
        os.mkdir(args.mask_path)
    if not os.path.exists(args.patch_path):
        os.mkdir(args.patch_path)
    ff = os.walk(args.wsi_path)
    paths = []
    opts_list = []
    for root, dirs, files in ff:
        for file in files:
            if os.path.splitext(file)[1] == '.svs':
                paths.append(os.path.join(root, file))
    
    print("---------------------------len(opts_list):",len(opts_list))
    pool = Pool(processes=args.num_process)
    pool.map(partial(process, args=args), paths)

    
       
           
            



def wsi_processing_check(coords,target_path):

    if len(coords)==len(os.listdir(target_path)):
        return True
    else:
        return False


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
