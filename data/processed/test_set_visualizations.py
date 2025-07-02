import os
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go

# CHANGRABLE VARIABLES
N = 2  # Change this to visualize different scenes
DATASET_DIR = r"D:\Datasets\MinimarketPointCloud\MiniMarket_point_clouds\test\all_test_scenes.h5"

def get_point_cloud(DATASET_DIR, N, percentage=1):
    with h5py.File(DATASET_DIR, 'r') as f:
        # Read datasets
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
    # valid_mask = ~np.isnan(point_clouds).any(axis=1) & ~np.isnan(color_clouds).any(axis=1)
    # point_clouds = point_clouds[valid_mask]
    # color_clouds = color_clouds[valid_mask]
    # label_clouds = label_clouds[valid_mask]
    return point_clouds, color_clouds, label_clouds


def plot_pc_rgb(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="RGB Point Cloud")

def plot_pc_labels(points, labels):
    colormap = np.array([[0, 1, 0], [1, 0, 0]])  # green, red
    label_colors = colormap[np.argmax(labels, axis=1)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    o3d.visualization.draw_geometries([pcd], window_name="Labelled Point Cloud")

def plot_pc_rgb_and_labels(points, colors, labels, offset=1.0):
    # Create original RGB point cloud
    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(points)
    pcd_rgb.colors = o3d.utility.Vector3dVector(colors)

    # Create labeled point cloud with a translation in X
    colormap = np.array([[0, 1, 0], [1, 0, 0]])  # green = object, red = background
    label_colors = colormap[np.argmax(labels, axis=1)]

    pcd_labels = o3d.geometry.PointCloud()
    translated_points = points.copy()
    translated_points[:, 0] += offset  # shift along X-axis
    pcd_labels.points = o3d.utility.Vector3dVector(translated_points)
    pcd_labels.colors = o3d.utility.Vector3dVector(label_colors)

    # Combine and show
    o3d.visualization.draw_geometries(
        [pcd_rgb, pcd_labels],
        window_name="Left: RGB | Right: Segmentation Labels"
    )

for N in range(0, 10):
    point_clouds, color_clouds, label_clouds = get_point_cloud(DATASET_DIR, N,0.1)
    # plot_pc_rgb(point_clouds, color_clouds)
    # plot_pc_labels(point_clouds, label_clouds)
    plot_pc_rgb_and_labels(point_clouds, color_clouds, label_clouds)
