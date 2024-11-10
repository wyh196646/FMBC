import sys
import os
import argparse
import logging
import glob
from tqdm import tqdm
import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save'
                                 ' it in npy format')
parser.add_argument('--wsi_path', default='/home/yuhaowang/data/raw_data/TCGA-Toy', metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('--mask_path', default='/home/yuhaowang/data/raw_data/TCGA-Toy-test-mask', metavar='GRAY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=2, type=int, help='at which WSI level'
                    ' to obtain the mask, default 6')
parser.add_argument('--RGB_min', default=50, type=int, help='min value for RGB'
                    ' channel, default 50')
parser.add_argument('--num_process', default=1, type=int,
                    help='number of mutli-process, default 5')

def getAllFile(img_dir, ext):
	filelist = []
	for root, dirs, files in os.walk(img_dir):
		for name in files:
			if os.path.splitext( name )[1] == ext:
				filelist.append(os.path.join(root, name))
	return filelist

        
        
def process(path,args=None):

    npy_name = os.path.basename(path)
    npy_path = os.path.join(args.mask_path,npy_name.replace('svs','npy'))

    if os.path.exists(npy_path):
        print(npy_path,'have been processed')
        return
    slide = openslide.OpenSlide(path)

    

    img_RGB = np.transpose(np.array(slide.read_region((0, 0),
                            min(args.level,slide.level_count-1),
                            slide.level_dimensions[min(args.level,slide.level_count-1)]).convert('RGB')),
                            axes=[1, 0, 2])

    slide.close()

    img_HSV = rgb2hsv(img_RGB)


    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    #print("444")
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    #print("555")
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    #print("666")
    min_R = img_RGB[:, :, 0] > args.RGB_min
    min_G = img_RGB[:, :, 1] > args.RGB_min
    min_B = img_RGB[:, :, 2] > args.RGB_min
    #print("777")

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    #tissue_mask = tissue_RGB & min_R & min_G & min_B
    #print("888")
    np.save(npy_path, tissue_mask)
    plt.imsave(os.path.join(args.mask_path, npy_name.replace('svs','png')), tissue_mask)
    print('slide', npy_name, 'done')
           
#Convert color space, OSTU threshold segmentation, open and close morphological operations
def run(args):
    logging.basicConfig(level=logging.INFO)

    #paths = glob.glob(os.path.join(args.wsi_path, '*.svs'))
    paths = getAllFile(args.wsi_path, '.svs')
    if not os.path.exists(args.mask_path):
        os.makedirs(args.mask_path)
    
    if len(os.path.join(args.mask_path, '*.npy')) == len(paths):
        return
    

    p = Pool(args.num_process)
    p.map(partial(process, args=args), paths)
  
    print('all_slide have been processed')  
    
    

def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
