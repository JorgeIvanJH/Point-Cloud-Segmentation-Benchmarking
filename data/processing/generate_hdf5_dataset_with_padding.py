import open3d as o3d
import os
import glob
import numpy as np
import h5py
import random



def generate_hdf5_dataset_with_padding(input_dir, output_hdf5_path,  num_points= 2048, seed = 42):
    np.random.seed(seed)

    pcd_files = sorted(glob.glob(os.path.join(input_dir, "*.pcd")))

    if not pcd_files:
        raise FileNotFoundError(f"No .pcd files found in {input_dir}")
    
    print(f"Found {len(pcd_files)} .pcd files.")
		
    point_clouds = []
    color_clouds = []

    for i, pcd_path in enumerate(pcd_files):
        cloud = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors) if cloud.has_colors() else np.zeros_like(points)

        n_points = points.shape[0]

        if n_points == 0:
            print(f"Skipping empty point cloud: {pcd_path} because it has no points.")
            continue

        if n_points > num_points:
            selected_idx = random.sample(range(n_points), num_points)
            points = points[selected_idx]
            colors = colors[selected_idx]

        elif n_points < num_points:
            pad_amt = num_points - n_points
            points = np.pad(points, ((0, pad_amt), (0, 0)), mode='constant')
            colors = np.pad(colors, ((0, pad_amt), (0, 0)), mode='constant')

        point_clouds.append(points)
        color_clouds.append(colors)

        print(f"Processed {i+1}/{len(pcd_files)}: {os.path.basename(pcd_path)} of object {os.path.basename(input_dir)}")
	
    # Save to HDF5
    print(f"Saving dataset to: {output_hdf5_path}")
    with h5py.File(output_hdf5_path, 'w') as f:
        f.create_dataset("point_clouds", data=np.stack(point_clouds), compression="gzip")
        f.create_dataset("color_clouds", data=np.stack(color_clouds), compression="gzip")

    print("finished")
