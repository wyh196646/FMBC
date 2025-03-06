import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from PIL import Image
from typing import Optional, Tuple


def create_overlay(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    scale: np.ndarray,
    region_size: Tuple[int, int]
) -> np.ndarray:
    """
    Create the heatmap overlay based on scores and coordinates.
    
    Args:
        scores (np.ndarray): Normalized scores.
        coords (np.ndarray): Coordinates of patches.
        patch_size_level0 (int): Patch size at level 0.
        scale (np.ndarray): Scaling factors.
        region_size (Tuple[int, int]): Dimensions of the region.
    
    Returns:
        np.ndarray: Heatmap overlay.
    """
    patch_size = np.ceil(np.array([patch_size_level0, patch_size_level0]) * scale).astype(int)
    coords = np.ceil(coords * scale).astype(int)
    
    overlay = np.zeros(tuple(np.flip(region_size)), dtype=float)
    counter = np.zeros_like(overlay, dtype=np.uint16)
    
    for idx, coord in enumerate(coords):
        overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += scores[idx]
        counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1
    
    zero_mask = counter == 0
    overlay[~zero_mask] /= counter[~zero_mask]
    overlay[zero_mask] = np.nan  # Set areas with no data to NaN
    
    return overlay


def apply_colormap(overlay: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Apply a colormap to the heatmap overlay.
    
    Args:
        overlay (np.ndarray): Heatmap overlay.
        cmap_name (str): Colormap name.

    Returns:
        np.ndarray: Colored overlay image.
    """
    cmap = plt.get_cmap(cmap_name)
    overlay_colored = np.zeros((*overlay.shape, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(overlay)
    colored_valid = (cmap(overlay[valid_mask]) * 255).astype(np.uint8)[:, :3]
    overlay_colored[valid_mask] = colored_valid
    return overlay_colored


def visualize_heatmap(
    wsi,
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    vis_level: Optional[int] = 2,
    cmap: str = 'coolwarm',
    normalize: bool = True
) -> Image.Image:
    """
    Generate a heatmap visualization overlayed on a whole slide image (WSI).
    
    Args:
        wsi: Whole slide image object.
        scores (np.ndarray): Scores associated with each coordinate.
        coords (np.ndarray): Coordinates of patches at level 0.
        patch_size_level0 (int): Patch size at level 0.
        vis_level (Optional[int]): Visualization level.
        cmap (str): Colormap to use for the heatmap.
        normalize (bool): Whether to normalize the scores.
    
    Returns:
        Image.Image: The final heatmap visualization.
    """
    if normalize:
        scores = rankdata(scores, 'average') / len(scores) * 100 / 100
    
    downsample = wsi.level_downsamples[vis_level]
    scale = np.array([1 / downsample, 1 / downsample])
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    
    overlay = create_overlay(scores, coords, patch_size_level0, scale, region_size)
    
    img = wsi.read_region_pil((0, 0), vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    img = np.array(img)
    
    overlay_colored = apply_colormap(overlay, cmap)
    blended_img = cv2.addWeighted(img, 0.6, overlay_colored, 0.4, 0)
    
    return Image.fromarray(blended_img)
