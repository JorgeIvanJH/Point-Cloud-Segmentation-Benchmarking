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


def generate_labelled_hdf5_testset(input_dir, output_hdf5_path):
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
        labels[np.all(labels == [1, 1], axis=1)] = [0, 1] # repeated to background
        labels[np.all(labels == [0, 0], axis=1)] = [0, 1] # repeated to object
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
    print("finished full labelled scenes generation")

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
    max_box = 3 # 3 meters
    tries = 0
    while num_points_in_box < max_points_in_box:
        tries += 1
        points_in_box_mask = np.all((points_sample >= box_min) & (points_sample <= box_max), axis=1)
        points_in_box = points_sample[points_in_box_mask]
        colors_in_box = colors_sample[points_in_box_mask]  # Make sure this is assigned
        labels_in_box = labels_sample[points_in_box_mask]  # Make sure this is assigned
        num_points_in_box = points_in_box.shape[0]
        
        if num_points_in_box < max_points_in_box:
            box_edge += increase_rate
            box_min = centroid - box_edge / 2
            box_max = centroid + box_edge / 2
            if box_edge > max_box:
                print(f"Warning: Box edge exceeded maximum size of {max_box} meters after {tries} tries. Stopping expansion.")
                break
        
        

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

def generate_augmented_testset(input_full_hdf5_dir, num_orientations=10, num_downsamplings = 10, range_downsampling=[0.1, 1.0], max_points_in_box=20480, increase_rate=0.001, scenes_included=None):
    """
    Generates an augmented test set by applying random transformations to the point clouds.
    """
    with h5py.File(input_full_hdf5_dir, "r") as f:
        point_clouds = f["seg_points"][:]
        color_clouds = f["seg_colors"][:]
        label_clouds = f["seg_labels"][:]
    if scenes_included is not None:
        point_clouds = point_clouds[scenes_included]
        color_clouds = color_clouds[scenes_included]
        label_clouds = label_clouds[scenes_included]

    B, num_points, _ = point_clouds.shape


    all_seg_sample_points = []
    all_seg_sample_colors = []
    all_seg_sample_labels = []
    count = 0
    for i in range(B):
        for j in range(num_orientations):
            for k in range(num_downsamplings):
                count += 1
                print(f"Processing point cloud {i+1}/{B}, orientation {j+1}/{num_orientations}, downsampling {k+1}/{num_downsamplings}")
                # Random rotation
                rotation_angle = np.random.uniform(0, 2 * np.pi)
                rotation_axis = np.random.uniform(-1, 1, size=3)
                rotation_axis /= np.linalg.norm(rotation_axis)
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

                # Apply rotation
                rotated_points = point_clouds[i] @ rotation_matrix.T
                rotated_colors = color_clouds[i]
                rotated_labels = label_clouds[i]

                # Random downsampling
                downsample_factor = np.random.uniform(range_downsampling[0], range_downsampling[1])
                num_downsampled_points = int(num_points * downsample_factor)
                if num_downsampled_points < num_points:
                    indices = np.random.choice(num_points, num_downsampled_points, replace=False)
                    downsampled_points = rotated_points[indices]
                    downsampled_colors = rotated_colors[indices]
                    downsampled_labels = rotated_labels[indices]
                else:
                    downsampled_points = rotated_points
                    downsampled_colors = rotated_colors
                    downsampled_labels = rotated_labels

                # Get voxel around the object of interest
                downsampled_points, downsampled_colors, downsampled_labels = get_voxel(
                    downsampled_points, downsampled_colors, downsampled_labels,
                    max_points_in_box=max_points_in_box, increase_rate=increase_rate
                )

                # Append to the lists
                all_seg_sample_points.append(downsampled_points)
                all_seg_sample_colors.append(downsampled_colors)
                all_seg_sample_labels.append(downsampled_labels)
                
                if count % 100 == 0:
                    print(f"Processed {count} samples so far.")
                    # Save the augmented point cloud
                    output_filename = f"augmentedscenes{scenes_included}_orientations{j+1}_downsamplings{k+1}_maxpointsinbox{max_points_in_box}.h5"
                    output_dir = os.path.join(os.path.dirname(input_full_hdf5_dir), output_filename)
                    print(f"Saving augmented dataset to: {output_dir}")
                    with h5py.File(output_dir, 'w') as f:
                        f.create_dataset("seg_points", data=np.asarray(all_seg_sample_points))
                        f.create_dataset("seg_colors", data=np.asarray(all_seg_sample_colors))
                        f.create_dataset("seg_labels", data=np.asarray(all_seg_sample_labels))
    print(f"Processed {count} samples so far.")
    # Save the augmented point cloud
    output_filename = f"augmentedscenes{scenes_included}_orientations{j+1}_downsamplings{k+1}_maxpointsinbox{max_points_in_box}.h5"
    output_dir = os.path.join(os.path.dirname(input_full_hdf5_dir), output_filename)
    print(f"Saving augmented dataset to: {output_dir}")
    with h5py.File(output_dir, 'w') as f:
        f.create_dataset("seg_points", data=np.asarray(all_seg_sample_points))
        f.create_dataset("seg_colors", data=np.asarray(all_seg_sample_colors))
        f.create_dataset("seg_labels", data=np.asarray(all_seg_sample_labels))


if __name__ == '__main__':

    # raw_input_dir = r"D:\Datasets\MinimarketPointCloud\MiniMarket_raw_test\unlabelled"
    # cleaned_input_dir = r"D:\Datasets\MinimarketPointCloud\MiniMarket_raw_test\unlabelled_clean" # UPLOAD THESE .pcd TO THE LABELLING WEB APP
    # clean_point_clouds(raw_input_dir, cleaned_input_dir) # CLEAN
    # input_labelled_dir = r"D:\Datasets\MinimarketPointCloud\MiniMarket_raw_test\labelled" # WHEN DOWNLOADING, STORE THE LABELLED .pcd AND .json FILES HERE
    # output_full_hdf5_dir = r"D:\Datasets\MinimarketPointCloud\MiniMarket_point_clouds\test" # HERE THE HDF5 FILE WILL BE SAVED
    # output_full_labelled_scene_filename = "all_test_scenes.h5"
    # output_full_hdf5_dir = os.path.join(output_full_hdf5_dir, output_full_labelled_scene_filename)
    # generate_labelled_hdf5_testset(input_labelled_dir, output_full_hdf5_dir) # GENERATE HDF5 DATASET WITH 
    input_full_hdf5_dir = r"D:\Datasets\MinimarketPointCloud\MiniMarket_point_clouds\test\all_test_scenes.h5"
    # MODIFIABLE PARAMETERS
    num_orientations = 10
    num_downsamplings = 10
    range_downsampling=[0.3, 0.5]
    max_points_in_box=100000
    increase_rate=0.001
    scenes_included = [0, 2, 3, 4, 5, 6, 7, 8] # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] Change this to include specific scenes
    generate_augmented_testset(input_full_hdf5_dir, num_orientations, num_downsamplings, range_downsampling, max_points_in_box, increase_rate, scenes_included)
