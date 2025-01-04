
#TCGA-AR-A1AL-01Z-00-DX1.h5

import numpy as np
from matplotlib import collections, patches, pyplot as plt
import pickle
from sklearn.cluster import KMeans
import h5py

def visualize_tile_locations(slide_sample, output_path, tile_info_list, tile_size, origin_offset):
    # check slide_image size. should be thumbnail size?
    slide_image = slide_sample["image"]
    downscale_factor = slide_sample["scale"]

    fig, ax = plt.subplots()
    ax.imshow(slide_image.transpose(1, 2, 0))
    rects = []
    for tile_info in tile_info_list:
        # change coordinate to the current level from level-0
        # tile location is in the original image cooridnate, while the slide image is after selecting ROI
        xy = ((tile_info["tile_x"] - origin_offset[0]) / downscale_factor,
              (tile_info["tile_y"] - origin_offset[1]) / downscale_factor)
        rects.append(patches.Rectangle(xy, tile_size, tile_size))
    pc = collections.PatchCollection(rects, match_original=True, alpha=0.5, edgecolor="black")
    pc.set_array(np.array([100] * len(tile_info_list)))
    ax.add_collection(pc)
    fig.savefig(output_path)
    plt.close()
 
def read_assets_from_h5( h5_path: str) -> tuple:
    assets = {}
    attrs = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            assets[key] = f[key][:]
            if f[key].attrs is not None:
                attrs[key] = dict(f[key].attrs)
    return assets, attrs
   
def kmeans_clustering(features, n_clusters = 8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

def visualize_tile_grid(slide_sample, output_path, tile_info_list, tile_size, origin_offset, cluster_labels, coords):
    """
    Visualizes the tile grid on a slide image with cluster-based colors.

    Args:
        slide_sample (dict): Slide image and scale information.
        output_path (str): Path to save the visualization.
        tile_info_list (list): List of tile information.
        tile_size (int): Size of the tiles.
        origin_offset (tuple): Offset for origin in x and y.
        cluster_labels (np.ndarray): Cluster labels for each coordinate.
        coords (np.ndarray): Array of coordinates corresponding to the tiles.
    """
    # Slide image and downscale factor
    slide_image = slide_sample["image"]
    downscale_factor = slide_sample["scale"]
    
    # Adjust the tile size to make rectangles larger and connected
    #expanded_tile_size = tile_size * 1.2  # 1.2倍扩大（可以根据需求调整）

    # Create a mapping of coordinates to cluster labels
    coords_cluster_map = {
        tuple(coord): cluster_labels[i] for i, coord in enumerate(coords)
    }
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(18, 18))  # 增加图像的显示尺寸
    ax.imshow(slide_image.transpose(1, 2, 0))
    
    # Color map for clusters
    num_classes = len(np.unique(cluster_labels))
    cmap = plt.cm.get_cmap('tab10', num_classes)  # Use a color map with `num_classes` colors
    
    # Process each tile
    for tile_info in tile_info_list:
        # Get the tile's coordinates
        tile_coord = (tile_info['tile_x'], tile_info['tile_y'])
        
        # Match the tile to its cluster label
        cluster_label = coords_cluster_map.get(tile_coord, None)
        if cluster_label is not None:
            # Calculate rectangle position for the scaled slide image
            x = (tile_info["tile_x"] - origin_offset[0]) / downscale_factor
            y = (tile_info["tile_y"] - origin_offset[1]) / downscale_factor
            
            # Draw the grid (outline) and fill with transparency
            rect = patches.Rectangle(
                (x, y),
                tile_size,
                tile_size,
                linewidth=1,
                edgecolor="black",  # Grid line color
                facecolor=cmap(cluster_label),  # Fill color from cluster
                alpha=0.7,  # 增加透明度
            )
            ax.add_patch(rect)
    
    # Save the figure with higher resolution
    fig.savefig(output_path, dpi=100)  # 使用更高的dpi保存
    plt.close()


n_clusters = 8  #
data, _ = read_assets_from_h5("/ruiyan/yuhao/embedding/TCGA-BRCA/TCGA-AR-A1AK-01Z-00-DX1.h5")
feature = data["features"]
coords= data["coords"]
cluster_labels = kmeans_clustering(feature, n_clusters)
num_classes = len(np.unique(cluster_labels))
colors = plt.cm.get_cmap('tab10', num_classes) 

sample = pickle.load(open("./sample.pkl", "rb"))
tile_info= pickle.load(open('./tile_info_list.pkl', "rb"))


# Process `tile_info` to match with `coords` and `cluster_labels`
for tile in tile_info:
    tile_coord = (tile["tile_x"], tile["tile_y"])
    if tile_coord in coords:
        index = np.where((coords == tile_coord).all(axis=1))[0]
        if len(index) > 0:
            tile["cluster_label"] = cluster_labels[index[0]]

# Update the call to `visualize_tile_grid` with the new arguments
visualize_tile_grid(
    sample,
    "./temp_roi_grid_colored.png",
    tile_info,
    256,
    origin_offset=sample["origin"],
    cluster_labels=cluster_labels,
    coords=coords
)
