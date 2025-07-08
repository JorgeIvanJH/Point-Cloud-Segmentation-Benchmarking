# 1. complete viz_utils.py to use it for predictions and labels and original samples.staticmethod
# 2. visualize both train/valid and test using viz_utils.py
# 3. put test.h5 to torchpoints3d and run and see predictions
# 4. r

import os
import glob
import random
import getpass
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import plotly.graph_objects as go
import sys
import open3d as o3d
import random
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import random

def get_point_cloud(DATASET_DIR, N, percentage=1):
    # Loads a point cloud from HDF5 file and samples a subset
    with h5py.File(DATASET_DIR, 'r') as f:
        point_clouds = f["seg_points"][:]
        color_clouds = f["seg_colors"][:]
        label_clouds = f["seg_labels"][:]
    
    B, num_points, _ = point_clouds.shape
    print("Number of clouds:", B, ", Chosen cloud index:", N)
    new_num_points = int(num_points * percentage)
    idx_downsample = random.sample(range(num_points), new_num_points)
    point_clouds = point_clouds[N, idx_downsample, :]
    color_clouds = color_clouds[N, idx_downsample, :]
    label_clouds = label_clouds[N, idx_downsample, :]
    
    return point_clouds, color_clouds, label_clouds


def plot_point_cloud(points, colors=None, labels=None):
    # Visualizes point cloud using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if labels is not None:
        label_colors = np.array([[0, 1, 0], [1, 0, 0]])[np.argmax(labels, axis=1)]
        pcd.colors = o3d.utility.Vector3dVector(label_colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd])


def plot_matplotlib_pc(points, colors=None, labels=None):
    # Visualizes point cloud using Matplotlib
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        label_colors = np.array([[0, 1, 0], [1, 0, 0]])[np.argmax(labels, axis=1)]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=label_colors, s=1)
        ax.set_title("Label Visualization")
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
        ax.set_title("Color Visualization")
    
    ax.set_axis_off()
    plt.show()


def visualize_predictions(model, weights_path, dataset_sample, threshold=0.5):
    model_weights = torch.load(weights_path)
    model.load_state_dict(model_weights)
    model.eval()
    dataset_samples_tensor = torch.tensor(dataset_sample, dtype=torch.float32).to("cpu")
    
    logits = model(dataset_samples_tensor)[0]
    output = (torch.softmax(logits, dim=2)[:, :, 0] > threshold).int()
    return output


def plot_pointcloud(points, colors = None, labels = None, predictions = None, method='open3d'):
    # Visualizes predicted labels compared with ground truth labels
    
    if method == 'open3d':
        plot_point_cloud(points, colors, labels)
        if predictions is not None:
            plot_point_cloud(points, colors, predictions)
    elif method == 'matplotlib':
        plot_matplotlib_pc(points, colors, labels)
        if predictions is not None:
            plot_matplotlib_pc(points, colors, predictions)

def get_voxel(points_sample, colors_sample, labels_sample, max_points_in_box=20480, increase_rate=0.001):
    assert points_sample.shape[0] == colors_sample.shape[0] == labels_sample.shape[0], "Points, colors, and labels must have the same number of points"
    assert points_sample.shape[0] >= max_points_in_box, "Number of points in the sample must be greater than or equal to max_points_in_box"
    # Remove points at the origin
    mask_not_in_origin = np.all(points_sample != np.array([0, 0, 0]), axis=1)
    points_sample, colors_sample, labels_sample = points_sample[mask_not_in_origin], colors_sample[mask_not_in_origin], labels_sample[mask_not_in_origin]

    # Select points belonging to the object (label [1, 0])
    object_idxs = np.all(labels_sample == np.array([1, 0]), axis=1)
    object_points = points_sample[object_idxs]
    centroid = np.mean(object_points, axis=0)
    
    num_points_in_box = 0
    box_edge = increase_rate
    box_min = centroid - box_edge / 2
    box_max = centroid + box_edge / 2
    
    # Expand box until we reach the desired number of points
    while num_points_in_box < max_points_in_box:
        points_in_box_mask = np.all((points_sample >= box_min) & (points_sample <= box_max), axis=1)
        points_in_box = points_sample[points_in_box_mask]
        colors_in_box = colors_sample[points_in_box_mask]  # Make sure this is assigned
        labels_in_box = labels_sample[points_in_box_mask]  # Make sure this is assigned
        num_points_in_box = points_in_box.shape[0]
        
        if num_points_in_box < max_points_in_box:
            box_edge += increase_rate
            box_min = centroid - box_edge / 2
            box_max = centroid + box_edge / 2

    # If fewer points, pad with zeros
    if num_points_in_box < max_points_in_box:
        padding_size = max_points_in_box - num_points_in_box
        points_padding = np.zeros((padding_size, points_sample.shape[1]))
        colors_padding = np.zeros((padding_size, colors_sample.shape[1]))
        labels_padding = np.zeros((padding_size, labels_sample.shape[1]))
        
        # Concatenate the original points with padding
        points_in_box = np.vstack((points_in_box, points_padding))
        colors_in_box = np.vstack((colors_in_box, colors_padding))
        labels_in_box = np.vstack((labels_in_box, labels_padding))

    # If more points, randomly sample to match max_points_in_box
    elif num_points_in_box > max_points_in_box:
        random_indices = np.random.choice(num_points_in_box, size=max_points_in_box, replace=False)
        points_in_box = points_in_box[random_indices]
        colors_in_box = colors_in_box[random_indices]  # Now correctly using colors_in_box
        labels_in_box = labels_in_box[random_indices]  # Now correctly using labels_in_box
        
    assert points_in_box.shape[0] == max_points_in_box, "Number of points in box does not match max_points_in_box"
    assert colors_in_box.shape[0] == max_points_in_box, "Number of colors in box does not match max_points_in_box"
    assert labels_in_box.shape[0] == max_points_in_box, "Number of labels in box does not match max_points_in_box"

    return points_in_box, colors_in_box, labels_in_box


# Sample visualization call
PC_SELECTED = 1 # np.random.randint(1000)  # Change this to visualize different scenes
percentage = 1 # Take a percentage of points, e.g., 25%
DATASET_DIR = r"D:\Datasets\MinimarketPointCloud\MiniMarket_point_clouds\2048\segmentation_dataset\REAL_BACKGROUND_ketchup_heinz_400ml_numPoints_20480_maxObjects_10_numscenes_100.h5"
# Load data
for PC_SELECTED in range(1000):
    print("PC_SELECTED: ", PC_SELECTED)
    points_sample, colors_sample, labels_sample = get_point_cloud(DATASET_DIR, PC_SELECTED, percentage)
    print("points_sample shape:", points_sample.shape)
    print("colors_sample shape:", colors_sample.shape)
    print("labels_sample shape:", labels_sample.shape)
    assert points_sample.shape[0] == colors_sample.shape[0] == labels_sample.shape[0] == 20480
    assert points_sample.shape[1] == 3, "Points should have 3 dimensions (x, y, z)"
    assert colors_sample.shape[1] == 3, "Colors should have 3 dimensions (r, g, b)"
    assert labels_sample.shape[1] == 2, "Labels should have 2 dimensions (target, alien)"
    # Visualize the original sample
    # plot_pointcloud(points_sample, colors=colors_sample)
    # plot_pointcloud(points_sample, labels= labels_sample)
