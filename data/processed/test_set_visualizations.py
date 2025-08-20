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

def create_grid(x_range, y_range, step=1.0, z_level=0.0, color=[0.7, 0.7, 0.7]):
    """
    Create a grid of lines in the XY plane at a fixed z_level.
    """
    lines = []
    points = []

    # Vertical lines (constant x)
    for x in np.arange(x_range[0], x_range[1] + step, step):
        points.append([x, y_range[0], z_level])
        points.append([x, y_range[1], z_level])
        lines.append([len(points) - 2, len(points) - 1])

    # Horizontal lines (constant y)
    for y in np.arange(y_range[0], y_range[1] + step, step):
        points.append([x_range[0], y, z_level])
        points.append([x_range[1], y, z_level])
        lines.append([len(points) - 2, len(points) - 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set

def plot_pc_rgb_and_labels(points, colors, labels, offset=1.0, grid_step=0.5):
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

    # Create grid around the two point clouds
    full_points = np.vstack([points, translated_points])
    min_bound = full_points.min(axis=0)
    max_bound = full_points.max(axis=0)

    # Add margin to bounds
    margin = 0.5
    x_range = (min_bound[0] - margin, max_bound[0] + margin)
    y_range = (min_bound[1] - margin, max_bound[1] + margin)

    grid = create_grid(x_range, y_range, step=grid_step, z_level=min_bound[2] - 0.01)

    # Visualize all
    o3d.visualization.draw_geometries(
        [pcd_rgb, pcd_labels, grid],
        window_name="Left: RGB | Right: Segmentation Labels with Grid"
    )

for N in range(0, 10):
    point_clouds, color_clouds, label_clouds = get_point_cloud(DATASET_DIR, N,0.1)
    # plot_pc_rgb(point_clouds, color_clouds)
    # plot_pc_labels(point_clouds, label_clouds)
    plot_pc_rgb_and_labels(point_clouds, color_clouds, label_clouds)
