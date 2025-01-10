from __future__ import division
from pathlib import Path
import hashlib
import os
import logging

import numpy as np
from numba import njit
import time
from concurrent import futures
from tqdm import tqdm
from typing import Dict, Tuple
import os



__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

supported_extensions = {'.svs', '.tif', '.vms', '.vmu',
                        '.ndpi', '.scn', '.mrxs', '.tiff', '.svslide', '.bif', '.qptiff'}

def test_wsidir_write_permissions(wsi_dir: Path):
    try:
        testfile = wsi_dir/f"test_{str(os.getpid())}.tmp"
        Path(testfile).touch()
    except PermissionError:
        logging.warning("No write permissions for wsi directory! If multiple stamp processes are running "
                        "in parallel, the final summary may show an incorrect number of slides processed.")
    finally:
        clean_lockfile(testfile)

def save_image(image, path: Path):
    width, height = image.size
    if width > 65500 or height > 65500:
        logging.warning(f"Image size ({width}x{height}) exceeds maximum size of 65500x65500, "
                        f"{path.name} will not be cached...")
        return
    image.save(path)
    
    
"""
Stain normalization based on the method of:

M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

@njit
def v1v2_mult(V, minPhi, maxPhi):
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    return v1, v2

def get_stain_matrix(I, beta=0.15, alpha=1):
    """
    Get stain matrix (2x3)
    :param I:
    :param beta:
    :param alpha:
    :return:
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = (OD[(OD > beta).any(axis=1), :])
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    That = np.dot(OD, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1, v2 = v1v2_mult(V, minPhi, maxPhi)

    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    # v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    # v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    # if v1[0] > v2[0]:
    #     HE = np.array([v1, v2])
    # else:
    #     HE = np.array([v2, v1])
    return normalize_rows(HE)


###
@njit
def transform_return(source_concentrations, stain_matrix_target, maxC_target, maxC_source, patch_shape):
    source_concentrations *= (maxC_target / maxC_source)
    return (255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target).reshape(patch_shape))).astype(
        np.uint8)



@njit
def hematoxalin_return(source_concentrations, h, w):
    H = source_concentrations[:, 0].reshape(h, w)
    H = np.exp(-1 * H)
    return H

def concurrent_concXstain(self, source_concentrations, patch_shapes, idx):
    maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
    maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
    jit_output = transform_return(source_concentrations, self.stain_matrix_target, maxC_target, maxC_source, patch_shapes[idx]) 
    return(jit_output)

from PIL import Image
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None

    def fit(self, target):
        target = standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)
        self.target_concentrations = get_concentrations_target(target, self.stain_matrix_target)

    def target_stains(self):
        return OD_to_RGB(self.stain_matrix_target)


    def transform(self, og_img: np.array, bg_rejected_img: np.array, rejected_list: np.array, patch_shapes: list, cores: int=8): #TODO: add optional split, patch sizes, overlap
        begin = time.time()
        stain_matrix_source = get_stain_matrix(og_img)
        after_sm = time.time()
        print(f"Get stain matrix: {after_sm-begin} seconds")
        I_shape = og_img.shape
        source_concentrations_list = get_concentrations_source(bg_rejected_img, I_shape, stain_matrix_source, rejected_list)
        after_conc = time.time()
        print(f"\nGet concentrations (normalisation): {after_conc-after_sm} seconds")

        del og_img, stain_matrix_source

        split=True
        if split:
            #added concurent concentrations x stain matrix
            with futures.ThreadPoolExecutor(cores) as executor:
                future_coords: Dict[futures.Future, int] = {}

                for i, source_concentrations in enumerate(source_concentrations_list):
                    # if all zeroes, skip
                    if np.any(source_concentrations):
                        future = executor.submit(concurrent_concXstain, self, source_concentrations=source_concentrations, patch_shapes=patch_shapes, idx=i)
                        future_coords[future] = i
                
                norm_img_patches_list = np.zeros((len(source_concentrations_list), 224, 224, 3), dtype=np.uint8)
                for tile_future in tqdm(futures.as_completed(future_coords), total=len(source_concentrations_list)-len(rejected_list), desc="Concentrations x Stain", leave=False):
                    i = future_coords[tile_future]
                    patch = tile_future.result()
                    norm_img_patches_list[i] = patch
                
            after_transform = time.time()
            print(f"\nConcentrations x Stain matrix: {after_transform-after_conc} seconds")

            print("Reconstructing image from patches...")
            norm_output_array = []
            canny_output_array = []
            for i in range(len(norm_img_patches_list)):
                # patch_shape = norm_img_patches_list[i].shape
                norm_output_array.append(np.array(norm_img_patches_list[i])) #removed .reshape(patch_shapes[i])
                if not rejected_list[i]:
                    canny_output_array.append(np.array(norm_img_patches_list[i])) #removed .reshape(patch_shapes[i])

            output_img = Image.new("RGB", (I_shape[1], I_shape[0]))
            canny_img = Image.new("RGB", (I_shape[1], I_shape[0]))

            coords_list=[]
            i_range = range(I_shape[0]//patch_shapes[0][0])
            j_range = range(I_shape[1]//patch_shapes[0][1])
            for i in i_range:
                for j in j_range:
                    idx = i*len(j_range) + j
                    output_img.paste(Image.fromarray(np.array(norm_output_array[idx])), (j*patch_shapes[idx][1], 
                                                                                i*patch_shapes[idx][0], 
                                                                                j*patch_shapes[idx][1]+patch_shapes[idx][1], 
                                                                                i*patch_shapes[idx][0]+patch_shapes[idx][0]))
                    canny_img.paste(Image.fromarray(np.array(bg_rejected_img[idx])), (j*patch_shapes[idx][1], 
                                                                                i*patch_shapes[idx][0], 
                                                                                j*patch_shapes[idx][1]+patch_shapes[idx][1], 
                                                                                i*patch_shapes[idx][0]+patch_shapes[idx][0]))
                    # print((j*patch_shapes[idx][1], 
                    #                                                             i*patch_shapes[idx][0], 
                    #                                                             j*patch_shapes[idx][1]+patch_shapes[idx][1], 
                    #                                                             i*patch_shapes[idx][0]+patch_shapes[idx][0]))
                    if not rejected_list[idx]:
                        coords_list.append((j*patch_shapes[idx][1], i*patch_shapes[idx][0]))


            #output_img = np.uint8(reconstruct_from_patches_2d(np.array(output), I_shape))
            del norm_img_patches_list
            # output_img = output_patches_list.reshape(1,-1)
            # breakpoint()
        
        else:
            maxC_source = np.percentile(source_concentrations_list, 99, axis=0).reshape((1, 2))
            maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
            jit_output = transform_return(source_concentrations_list, self.stain_matrix_target, maxC_target, maxC_source, I_shape) #I_shape, @3 (removed)
            after_transform = time.time()
            print(f"Concentrations x Stain matrix: {after_transform-after_conc} seconds")
            output_img = Image.fromarray(np.array(jit_output))
            output_array = jit_output
            coords_list = None #TODO

        return canny_img, output_img, canny_output_array, coords_list


    def hematoxylin(self, I):
        I = standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = get_concentrations_target(I, stain_matrix_source) #put target here, just in case

        del I
        del stain_matrix_source

        jit_output = hematoxalin_return(source_concentrations, h, w)

        return jit_output

#test get_stain_matrix
def test_get_stain_matrix():
    img = Image.open("test_images/1.jpg")
    img = np.array(img)
    img = standardize_brightness(img)
    stain_matrix = get_stain_matrix(img)
    assert stain_matrix.shape == (2,3)
    assert np.allclose(stain_matrix, [[0.650, 0.072, 0.740], [0.704, 0.990, 0.080]], atol=0.01)



"""
Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""


import numpy as np
import cv2 as cv
# import spams
import matplotlib.pyplot as plt
from numba import njit
import time
import logging


##########################################

def read_image(path):
    """
    Read an image to RGB uint8
    :param path:
    :return:
    """
    im = cv.imread(path)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im


def show_colors(C):
    """
    Shows rows of C as colors (RGB)
    :param C:
    :return:
    """
    n = C.shape[0]
    for i in range(n):
        if C[i].max() > 1.0:
            plt.plot([0, 1], [n - 1 - i, n - 1 - i], c=C[i] / 255, linewidth=20)
        else:
            plt.plot([0, 1], [n - 1 - i, n - 1 - i], c=C[i], linewidth=20)
        plt.axis('off')
        plt.axis([0, 1, -1, n])


def show(image, now=True, fig_size=(10, 10)):
    """
    Show an image (np.array).
    Caution! Rescales image to be in range [0,1].
    :param image:
    :param now:
    :param fig_size:
    :return:
    """
    image = image.astype(np.float32)
    m, M = image.min(), image.max()
    if fig_size != None:
        plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1])
    plt.imshow((image - m) / (M - m), cmap='gray')
    plt.axis('off')
    if now == True:
        plt.show()


def build_stack(tup):
    """
    Build a stack of images from a tuple of images
    :param tup:
    :return:
    """
    N = len(tup)
    if len(tup[0].shape) == 3:
        h, w, c = tup[0].shape
        stack = np.zeros((N, h, w, c))
    if len(tup[0].shape) == 2:
        h, w = tup[0].shape
        stack = np.zeros((N, h, w))
    for i in range(N):
        stack[i] = tup[i]
    return stack


def patch_grid(ims, width=5, sub_sample=None, rand=False, save_name=None):
    """
    Display a grid of patches
    :param ims:
    :param width:
    :param sub_sample:
    :param rand:
    :return:
    """
    N0 = np.shape(ims)[0]
    if sub_sample == None:
        N = N0
        stack = ims
    elif sub_sample != None and rand == False:
        N = sub_sample
        stack = ims[:N]
    elif sub_sample != None and rand == True:
        N = sub_sample
        idx = np.random.choice(range(N), sub_sample, replace=False)
        stack = ims[idx]
    height = np.ceil(float(N) / width).astype(np.uint16)
    plt.rcParams['figure.figsize'] = (18, (18 / width) * height)
    plt.figure()
    for i in range(N):
        plt.subplot(height, width, i + 1)
        im = stack[i]
        show(im, now=False, fig_size=None)
    if save_name != None:
        plt.savefig(save_name)
    plt.show()


######################################

def standardize_brightness(I):
    """

    :param I:
    :return:
    """
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def remove_zeros(I):
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I


def RGB_to_OD(I):
    """
    Convert from RGB to optical density
    :param I:
    :return:
    """
    I = remove_zeros(I)
    return -1 * np.log(I / 255)


def OD_to_RGB(OD):
    """
    Convert from optical density to RGB
    :param OD:
    :return:
    """
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def notwhite_mask(I, thresh=0.8):
    """
    Get a binary mask where true denotes 'not white'
    :param I:
    :param thresh:
    :return:
    """
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)


def sign(x):
    """
    Returns the sign of x
    :param x:
    :return:
    """
    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0



@njit
def transform_return(source_concentrations, stain_matrix_target, maxC_target, maxC_source):
    source_concentrations *= (maxC_target / maxC_source)
    # return (255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target).reshape(I_shape))).astype(
    #     np.uint8)
    return (255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target))).astype( #removed reshape, should be right format already
        np.uint8)



def get_concentrations_target(I, stain_matrix, lamda=0.01):
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """
    OD = RGB_to_OD(I).reshape((-1, 3))
    try:
        #limited Lasso to 1 thread, instead of taking all available threads (-1 default)
        temp, _, _, _ = np.linalg.lstsq(stain_matrix.T, OD.T, rcond=None)
        temp=temp.T
    except Exception as e:
        print(e)
        temp = None
    return temp

# def _get_concentrations_source_patch(patch, stain_matrix, patch_list, order_list, idx, lamda=0.01):
#     OD = RGB_to_OD(patch).reshape((-1, 3)) #.astype('float32') #change from float64 to float32, half memory

#     try:
#         temp = spams.lasso(OD.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T 
#     except Exception as e:
#         print(e)
#         temp = None

#     patch_list.append(temp) #save into list to unpatchify later

#     #clean up memory just in case
#     del temp
#     del OD

#     return temp


from concurrent import futures
from tqdm import tqdm
from typing import Dict, Tuple
import os

def get_concentrations_source(I, I_shape, stain_matrix, rejection_list, lamda=0.01):
    """
    Split the image I into big patches, loop over them, to OD + reshape, norm, reshape to I
    and in the end stitch the big patches together for the entire image again

    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """

    # logging.basicConfig(filename='norm-log.txt')
    # print = logging.info
    # patchify returns a NumPy array with shape (n_rows, n_cols, 1, H, W, N), if image is N-channels.
    # H W N is Height Width N-channels of the extracted patch
    # n_rows is the number of patches for each column and n_cols is the number of patches for each row

    if True: #(I_shape[0] + I_shape[1]) > (224*2): #bigger than 30k edge pixels combined, i.e. 15k x 15k
        #x = 500 # 2 for largest possible blocks
        x=(I_shape[0]//224)*(I_shape[1]//224)
        #print(f'Splitting WSI into {x*x} for normalisation...')
        print(f"Normalising {np.sum(~rejection_list)} tiles...")
        # print("Going into RGB->OD and spams Lasso function...")
        patches_shape = (224, 224) #(I_shape[0]//x, I_shape[1]//x)
        # patches = []

        patch_list =[]
        begin_time_list = []
	    #changed maximum threads from 32 to os.cpu_count()
        if os.cpu_count() > 8:
            cores = 8
        else:
            cores = os.cpu_count()
        with futures.ThreadPoolExecutor(cores) as executor: #os.cpu_count()
            future_coords: Dict[futures.Future, int] = {}
            i_range = range(I_shape[0]//patches_shape[0])
            j_range = range(I_shape[1]//patches_shape[1])
            for i in i_range:
                for j in j_range:          
                    patch = I[i*len(j_range) + j] #I[(i*patches_shape[0]):(i*patches_shape[0]+patches_shape[0]), (j*patches_shape[1]):(j*patches_shape[1]+patches_shape[1])]
                    #if rejected, just skip the patch
                    if not rejection_list[i*len(j_range) + j]:
                        future = executor.submit(
                            get_concentrations_target, patch, stain_matrix)
                        #print(f'Submitted patch #{2*i+j} into thread...')
                        begin_time_list.append(time.time())
                        future_coords[future] = i*len(j_range) + j # index 0 - 3. (0,0) = 0, (0,1) = 1, (1,0) = 2, (1,1) = 3

            #patch_list = np.zeros((x*x, I_shape[0]//x*I_shape[1]//x, 2), dtype=np.float64)
            patch_list = np.zeros((x, 224*224, 2), dtype=np.float64)
            for tile_future in tqdm(futures.as_completed(future_coords), total=np.sum(~rejection_list), desc='Normalising tiles', leave=False):
                i = future_coords[tile_future]
                #print(f'Received normalised patch #{i} from thread in {time.time()-begin_time_list[i]} seconds')
                patch = tile_future.result()
                patch_list[i] = patch
        
        del I

        return patch_list #len(patches_shapes_list)
    
    else:
        print('Normalising WSI as a whole...')
        split=False
        begin = time.time()
        print("Going into RGB->OD and spams Lasso function...")
        temp = get_concentrations_target(I, stain_matrix)
        end = time.time()
        print(f"Finished RGB->OD and spams Lasso function: {end-begin}")

        return None #temp, None, None, split


