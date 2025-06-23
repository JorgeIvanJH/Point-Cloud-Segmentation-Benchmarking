import os
import open3d as o3d
import glob
import h5py
import json
import numpy as np

def clean_point_clouds(input_dir, output_dir):
    print("Cleaning point clouds in directory:", input_dir)
    pcd_files = sorted(glob.glob(os.path.join(input_dir, "*.pcd")))

    for i, pcd_path in enumerate(pcd_files):
        print(f"Processing {i+1}/{len(pcd_files)}: {os.path.basename(pcd_path)}")
        cloudname = os.path.basename(pcd_path).split('.')[0]
        # Step 1: Load and remove NaN/inf
        cloud = o3d.io.read_point_cloud(pcd_path, remove_nan_points=True, remove_infinite_points=True)

        # Step 2: Remove zero or near-zero points
        cloud = cloud.select_by_index(np.where(np.linalg.norm(np.asarray(cloud.points), axis=1) > 1e-6)[0])

        # Step 3: Remove statistical outliers (e.g., noise) TODO: READ, DOCUMENT, AND CONSIDER ADDING
        # cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Step 4: (Optional) Remove radius outliers TODO: READ, DOCUMENT, AND CONSIDER ADDING
        # cloud, _ = cloud.remove_radius_outlier(nb_points=16, radius=0.05)

        # Step 5: (Optional) Downsample (voxel grid) TODO: READ, DOCUMENT, AND CONSIDER ADDING
        # cloud = cloud.voxel_down_sample(voxel_size=0.01)

        # Step 6: (Optional) Normalize to center and scale
        points = np.asarray(cloud.points)
        points -= points.mean(axis=0)
        # points /= np.linalg.norm(points, axis=1).max()  TODO: READ, DOCUMENT, AND CONSIDER ADDING
        cloud.points = o3d.utility.Vector3dVector(points)

        # Step 7: Save cleaned point cloud
        out_path = os.path.join(output_dir, f"cleaned_{cloudname}.pcd")
        o3d.io.write_point_cloud(out_path, cloud)

def pad_to_max_length(arrays, fill_value=0.0):
    max_len = max(arr.shape[0] for arr in arrays)
    padded = []
    for arr in arrays:
        pad_len = max_len - arr.shape[0]
        padded_arr = np.pad(arr, ((0, pad_len), (0, 0)), mode='constant', constant_values=fill_value)
        padded.append(padded_arr)
    return np.stack(padded)


def generate_hdf5_dataset(input_dir, output_hdf5_path):
    """
    Likely to be used to process TESTING data, where padding is not required.
    """
    pcd_files = sorted(glob.glob(os.path.join(input_dir, "*.pcd")))
    annotation_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    
    print(f"Found {len(pcd_files)} .pcd files.")
    print(f"Found {len(annotation_files)} .json files.")


		
    point_clouds = []
    color_clouds = []
    label_clouds = []

    for i, pcd_path in enumerate(pcd_files):
        cloudname = os.path.basename(pcd_path)
        annotation_path = [f for f in annotation_files if cloudname in f][0]

        cloud = o3d.io.read_point_cloud(pcd_path)
        annotation = json.load(open(annotation_path, 'r'))

        id_object = [obj["id"] for obj in annotation["objects"] if obj["classTitle"] == "object"][0]
        indices_object = [fig for fig in annotation["figures"] if fig["objectId"] == id_object][0]["geometry"]["indices"] # list of indices of the points that belong to the object
        indices_background = [fig for fig in annotation["figures"] if fig["objectId"] != id_object][0]["geometry"]["indices"] # list of indices of the points that belong to the background

        print("length of indices_object: ", len(indices_object))
        print("length of indices_background: ", len(indices_background))
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors) if cloud.has_colors() else np.zeros_like(points)
        n_points = points.shape[0]
        print("Number of points in point cloud: ", n_points)
        labels = np.zeros((n_points,2))
        labels[indices_background, 1] = 1
        labels[indices_object, 0] = 1
        labels = np.where(labels == [1,1], [0,1], labels) # repeated to background
        # labels = np.where(labels == [0,0], [0,1], labels) # repeated to object
        point_clouds.append(points)
        color_clouds.append(colors)
        label_clouds.append(labels)
        print("points shape: ", points.shape)
        print("colors shape: ", colors.shape)
        print("labels shape: ", labels.shape)
        print(f"Processed {i+1}/{len(pcd_files)}: {os.path.basename(pcd_path)}")

    # Save to HDF5
    print(f"Saving dataset to: {output_hdf5_path}")
    with h5py.File(output_hdf5_path, 'w') as f:
        f.create_dataset("seg_points", data=pad_to_max_length(point_clouds), compression="gzip")
        f.create_dataset("seg_colors", data=pad_to_max_length(color_clouds), compression="gzip")
        f.create_dataset("seg_labels", data=pad_to_max_length(label_clouds), compression="gzip")


    print("finished")

if __name__ == '__main__':

    input_dir = r"D:\Datasets\MinimarketPointCloud\MiniMarket_raw_test\unlabelled"
    cleaned_input_dir = r"D:\Datasets\MinimarketPointCloud\MiniMarket_raw_test\unlabelled_clean"
    # clean_point_clouds(input_dir, cleaned_input_dir) # CLEAN
    input_labelled_dir = r"D:\Datasets\MinimarketPointCloud\MiniMarket_raw_test\labelled"
    output_hdf5_path = r"D:\Datasets\MinimarketPointCloud\MiniMarket_point_clouds\test"
    output_filename = "all_test_scenes.h5"
    output_hdf5_path = os.path.join(output_hdf5_path, output_filename)
    generate_hdf5_dataset(input_labelled_dir, output_hdf5_path) # GENERATE HDF5 DATASET