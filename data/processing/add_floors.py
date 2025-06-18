import os
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go



# CHANGRABLE VARIABLES
DATASET_DIR = r"D:\Datasets\MinimarketPointCloud\MiniMarket_point_clouds\2048\segmentation_dataset\ketchup_heinz_400ml_segmentation_20250526_121710_numPoints_2048_maxObjects_10_orientations_1.h5"



filename = os.path.basename(DATASET_DIR)
path_before_filename = os.path.dirname(DATASET_DIR)
DATASET_OUTPUT_DIR = os.path.join(path_before_filename, "FLOORED_" + filename)

def get_point_cloud(DATASET_DIR, N):
    with h5py.File(DATASET_DIR, 'r') as f:
        # Read datasets
        seg_points = f["seg_points"][:]  
        seg_colors = f["seg_colors"][:]  
        seg_labels = f["seg_labels"][:]  
    print(seg_points.shape)
    print(seg_colors.shape)
    print(seg_labels.shape)
    
    print("Random sample index:", N)
    points_sample = seg_points[N, :, :]
    colors_sample = seg_colors[N, :, :]
    labels_sample = seg_labels[N, :, :]
    return points_sample, colors_sample, labels_sample

def get_all_point_clouds(DATASET_DIR):
    with h5py.File(DATASET_DIR, 'r') as f:
        # Read datasets
        seg_points = f["seg_points"][:]  
        seg_colors = f["seg_colors"][:]  
        seg_labels = f["seg_labels"][:]  
    print(seg_points.shape)
    print(seg_colors.shape)
    print(seg_labels.shape)
    
    return seg_points, seg_colors, seg_labels

def matplotlib_pc(points_sample, colors_sample, labels_sample):

    labels_sample = np.where((labels_sample == np.array([1, 0])).all(axis=1, keepdims=True), [0, 1, 0], [1, 0, 0])

    fig = plt.figure(figsize=(12, 6))

    # First subplot for color visualization
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2],
                c=colors_sample, s=1)
    ax1.set_title("Color Visualization")
    ax1.set_axis_off()

    # Second subplot for label visualization
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2],
                c=labels_sample, s=1)
    ax2.set_title("Label Visualization")
    ax2.set_axis_off()

    plt.show()

def open3d_pc(points_sample, colors_sample, labels_sample):
    pcd_colors = o3d.geometry.PointCloud()
    pcd_colors.points = o3d.utility.Vector3dVector(points_sample)
    pcd_colors.colors = o3d.utility.Vector3dVector(colors_sample)
    o3d.visualization.draw_geometries([pcd_colors], window_name="Color Visualization")

    pcd_labels = o3d.geometry.PointCloud()
    pcd_labels.points = o3d.utility.Vector3dVector(points_sample)
    pcd_labels.colors = o3d.utility.Vector3dVector(labels_sample)
    o3d.visualization.draw_geometries([pcd_labels], window_name="Label Visualization")

def add_floor(points_sample, colors_sample, labels_sample, num_floor_points=2000):
    # Step 1: Get valid object points
    object_mask = (labels_sample[:, 0] == 1)
    valid_mask = ~np.all(points_sample == 0, axis=1)
    object_points = points_sample[object_mask & valid_mask]

    is_colinear = True
    while is_colinear:
        # Step 2: Select 3 random, well-spaced points to define a floor plane
        indices = np.random.choice(object_points.shape[0], size=3, replace=False)
        p1, p2, p3 = object_points[indices]

        # Step 3: Define plane basis using p1 as origin
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) == 0:
            print("Chosen points are colinear; trying again.")
            continue
        is_colinear = False
        normal = normal / np.linalg.norm(normal)

    # Step 4: Create a grid of points in the plane (u, v are basis directions)
    resolution = 200  # Generate a large enough grid to sample from
    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)

    grid_u = np.linspace(-0.1, 0.1, resolution)
    grid_v = np.linspace(-0.1, 0.1, resolution)
    uu, vv = np.meshgrid(grid_u, grid_v)

    full_floor_points = p1 + uu[..., np.newaxis]*u + vv[..., np.newaxis]*v
    full_floor_points = full_floor_points.reshape(-1, 3)

    # Step 5: Randomly sample from the generated floor grid
    total_floor = full_floor_points.shape[0]
    num_samples = min(num_floor_points, total_floor)
    sample_indices = np.random.choice(total_floor, size=num_samples, replace=False)
    floor_points = full_floor_points[sample_indices]

    # Step 6: Create corresponding colors and labels
    floor_colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (num_samples, 1))
    floor_labels = np.tile(np.array([[0., 1.]]), (num_samples, 1))  # clutter

    # Step 7: Concatenate to original arrays
    points_augmented = np.vstack((points_sample, floor_points))
    colors_augmented = np.vstack((colors_sample, floor_colors))
    labels_augmented = np.vstack((labels_sample, floor_labels))

    return points_augmented, colors_augmented, labels_augmented


# points_sample, colors_sample, labels_sample = get_point_cloud(DATASET_DIR, 1)
# points_augmented, colors_augmented, labels_augmented = add_floor(points_sample, colors_sample, labels_sample)

# print("points_augmented shape:", points_augmented.shape)
# matplotlib_pc(points_augmented, colors_augmented, labels_augmented)


all_points_sample, all_colors_sample, all_labels_sample = get_all_point_clouds(DATASET_DIR)
print("All points shape:", all_points_sample.shape)


all_seg_sample_points = []
all_seg_sample_colors = []
all_seg_sample_labels = []

for points_sample, colors_sample, labels_sample in zip(all_points_sample, all_colors_sample, all_labels_sample):
    points_augmented, colors_augmented, labels_augmented = add_floor(points_sample, colors_sample, labels_sample)

    all_seg_sample_points.append(points_augmented)
    all_seg_sample_colors.append(colors_augmented)
    all_seg_sample_labels.append(labels_augmented)
    

with h5py.File(DATASET_OUTPUT_DIR, 'w') as f:
    f.create_dataset("seg_points", data=np.asarray(all_seg_sample_points))
    f.create_dataset("seg_colors", data=np.asarray(all_seg_sample_colors))
    f.create_dataset("seg_labels", data=np.asarray(all_seg_sample_labels))




points_sample, colors_sample, labels_sample = get_point_cloud(DATASET_OUTPUT_DIR, 0)
matplotlib_pc(points_sample, colors_sample, labels_sample)