import os
import numpy as np
import h5py
import random
import glob
import math
import datetime
import getpass

# TODO: Set seed
CONFIG_PATH = "../../config.yaml"

# Get current timestamp
timestamp = datetime.datetime.now().strftime("date_%Y%m%d_time_%H%M%S")

# Define Rotation Functions
def Rotx(t):
    return np.matrix([[1, 0, 0, 0], [0, np.cos(t), -np.sin(t), 0], [0, np.sin(t), np.cos(t), 0], [0, 0, 0, 1]])

def Roty(t):
    return np.matrix([[np.cos(t), 0, np.sin(t), 0], [0, 1, 0, 0], [-np.sin(t), 0, np.cos(t), 0], [0, 0, 0, 1]])

def Rotz(t):
    return np.matrix([[np.cos(t), -np.sin(t), 0, 0], [np.sin(t), np.cos(t), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def random_homogeneous_3d_rotation(x_min, x_max, y_min, y_max, z_min, z_max):
    angle_about_x = random.uniform(x_min, x_max)
    angle_about_y = random.uniform(y_min, y_max)
    angle_about_z = random.uniform(z_min, z_max)
    return Rotx(angle_about_x*math.pi/180) * Roty(angle_about_y*math.pi/180) * Rotz(angle_about_z*math.pi/180)

def random_homogeneous_3d_translation(x_min, x_max, y_min, y_max, z_min, z_max):
    distance_along_x = random.uniform(x_min, x_max)
    distance_along_y = random.uniform(y_min, y_max)
    distance_along_z = random.uniform(z_min, z_max)
    return np.matrix([
        [1, 0, 0, distance_along_x],
        [0, 1, 0, distance_along_y],
        [0, 0, 1, distance_along_z],
        [0, 0, 0, 1]
    ])

# Homogeneous transformation application function
def apply_random_transformation(points, num_points):
    points = np.concatenate((points, np.ones((num_points, 1))), axis=1)  # Add homogeneous coordinate
    transformation = random_homogeneous_3d_translation(-0.25, 0.25, -0.25, 0.25, 0.0, 0.25) * random_homogeneous_3d_rotation(-180, 180, -180, 180, -180, 180)
    transformed_points = np.matmul(transformation, np.transpose(points))
    transformed_points = np.transpose(transformed_points)
    return np.delete( transformed_points, 3, 1 )



# Main function to generate the object segmentation dataset
def generate_object_segmentation_dataset(objects_dataset_path, target_object_filename, generated_dataset_path, data_config):
    """
    objects_dataset_path: Path to the directory containing the processed dataset files with specific number of points per object (HDF5 files).
    target_object_filename: The filename of the target object HDF5 file (e.g., "object_1.h5").
    generated_dataset_path: Path to the directory where the generated segmentation dataset will be saved.
    max_num_objects: Maximum number of objects (including the target object) in each segmentation sample.
    num_orientations: Number of different random transformations to generate samples with

    note: the one-hot encoding of the labels is as follows:
        one-hot encoding [1, 0]  (Object of interest)
        one-hot encoding [0, 1]  (Background/clutter)

    """
    print("Generated Dataset Path: ", generated_dataset_path)

    max_num_objects  = data_config["MAX_NUM_OBJECTS"]
    min_num_objects = data_config["MIN_NUM_OBJECTS"]
    num_orientations = data_config["NUM_ORIENTATIONS"]

    if data_config["CHECKPOINT_DIR"]:
        CHECKPOINT_DIR = data_config["CHECKPOINT_DIR"]
        with h5py.File(CHECKPOINT_DIR, 'r') as f:
            point_clouds = f["seg_points"][:]
            color_clouds = f["seg_colors"][:]
            label_clouds = f["seg_labels"][:]
        all_seg_sample_points = [pc for pc in point_clouds]
        all_seg_sample_colors = [cc for cc in color_clouds]
        all_seg_sample_labels = [lc for lc in label_clouds]
        num_samples = len(all_seg_sample_points)
    else:
        all_seg_sample_points = []
        all_seg_sample_colors = []
        all_seg_sample_labels = []
        num_samples = 0

    # Get files
    target_hdf5_file = os.path.join(objects_dataset_path, target_object_filename)
    alien_hdf5_files = sorted(glob.glob(os.path.join(objects_dataset_path, "*.h5")))
    alien_hdf5_files.remove(target_hdf5_file)

    # Read the target object data
    with h5py.File(target_hdf5_file, "r") as f:
        target_points = f["point_clouds"][()]  # numpy array
        target_colors = f["color_clouds"][()]  # numpy array

    num_points_per_object = target_points.shape[1]
    num_samples_per_object = target_points.shape[0]
    NUM_POINTS_PER_SEG_SAMPLE = num_points_per_object * max_num_objects

    # Iterate through orientations
    for it in range(num_orientations):
        # Process each target sample
        for ts in range(num_samples_per_object):

            num_samples += 1
            print(f"Orientation transformation {it+1} / {num_orientations}. Processing target sample {ts + 1} / {num_samples_per_object}")
            target_sample_index = random.randrange(num_samples_per_object)  # Randomly select a target sample index
            selected_target_sample_point = target_points[target_sample_index, :, :]
            selected_target_sample_color = target_colors[target_sample_index, :, :]

            # Apply random transformation to target object
            selected_target_sample_point = apply_random_transformation(selected_target_sample_point, num_points_per_object)

            # Prepare segmentation sample (target object)
            seg_sample_point = selected_target_sample_point
            seg_sample_color = selected_target_sample_color
            seg_sample_label = np.concatenate((np.ones((num_points_per_object, 1)), np.zeros((num_points_per_object, 1))), axis=1)  # One-hot encoding [1, 0]

            # Random number of alien objects
            NUM_ALIEN_OBJECTS = random.randrange(min_num_objects, max_num_objects) - 1
            print(f"Processing sample {target_sample_index}, Number of alien objects: {NUM_ALIEN_OBJECTS}")

            # Randomly select alien object files
            copy_alien_files = alien_hdf5_files.copy()
            while len(copy_alien_files) > NUM_ALIEN_OBJECTS:
                copy_alien_files.pop(random.randrange(len(copy_alien_files)))

            # Add alien objects to the segmentation sample
            for i, alien_object_file in enumerate(copy_alien_files):
                with h5py.File(alien_object_file, "r") as f:
                    alien_points = f["point_clouds"][()]
                    alien_colors = f["color_clouds"][()]

                    # Random alien sample selection
                    alien_sample_index = random.randrange(alien_points.shape[0])
                    selected_alien_sample_point = alien_points[alien_sample_index, :, :]
                    selected_alien_sample_color = alien_colors[alien_sample_index, :, :]

                    # Apply random transformation to alien object
                    selected_alien_sample_point = apply_random_transformation(selected_alien_sample_point, num_points_per_object)

                    # Add alien data to segmentation sample
                    seg_sample_point = np.concatenate((seg_sample_point, selected_alien_sample_point), axis=0)
                    seg_sample_color = np.concatenate((seg_sample_color, selected_alien_sample_color), axis=0)
                    seg_sample_label = np.concatenate((seg_sample_label, np.concatenate((np.zeros((num_points_per_object, 1)), np.ones((num_points_per_object, 1))), axis=1)), axis=0) # One-hot encoding [0, 1] for alien objects

            # Pad the remaining sample size with zeros
            seg_sample_point = np.concatenate((seg_sample_point, np.zeros((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_point.shape[0], 3))), axis=0)
            seg_sample_color = np.concatenate((seg_sample_color, np.zeros((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_color.shape[0], 3))), axis=0)
            seg_sample_label = np.concatenate((seg_sample_label, np.concatenate((np.zeros(((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_label.shape[0]), 1)), np.ones(((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_label.shape[0]), 1))), axis=1)), axis=0) # Padding with [0, 1] for alien objects

            # Add the current segmentation sample to the dataset
            all_seg_sample_points.append(seg_sample_point)
            all_seg_sample_colors.append(seg_sample_color)
            all_seg_sample_labels.append(seg_sample_label)
            
            if num_samples % 100 == 0:
                print(f"Processed {num_samples} samples so far.")
                # Save the dataset
                hdf5_filename = f"{generated_dataset_path}_segmentation_{timestamp}_numPoints_{num_points_per_object}_minObjects_{min_num_objects}_maxObjects_{max_num_objects}_numrientations_{num_orientations}.h5"
                with h5py.File(hdf5_filename, 'w') as f:
                    f.create_dataset("seg_points", data=np.asarray(all_seg_sample_points))
                    f.create_dataset("seg_colors", data=np.asarray(all_seg_sample_colors))
                    f.create_dataset("seg_labels", data=np.asarray(all_seg_sample_labels))
    print("Dataset generation complete!")
    return

def get_voxel(points_sample, colors_sample, labels_sample, max_points_in_box=20480, increase_rate=0.001, centered_in_object=True):
    assert points_sample.shape[0] == colors_sample.shape[0] == labels_sample.shape[0], "Points, colors, and labels must have the same number of points"
    assert points_sample.shape[0] >= max_points_in_box, "Number of points in the sample must be greater than or equal to max_points_in_box"
    # Remove points at the origin
    mask_not_in_origin = np.all(points_sample != np.array([0, 0, 0]), axis=1)
    points_sample, colors_sample, labels_sample = points_sample[mask_not_in_origin], colors_sample[mask_not_in_origin], labels_sample[mask_not_in_origin]

    
    if centered_in_object: # Center the box in the object
        object_idxs = np.all(labels_sample == np.array([1, 0]), axis=1)
        object_points = points_sample[object_idxs]
        centroid = np.mean(object_points, axis=0)
    else: # The centroid is any random point in the sample
        rand_point_index = np.random.randint(0, points_sample.shape[0])
        centroid = points_sample[rand_point_index, :]

    
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

def downsampling(seg_sample_point, seg_sample_color, seg_sample_label, range_downsampling = (0.1, 0.5)):
    
    num_points = seg_sample_point.shape[0]
    downsample_factor = np.random.uniform(range_downsampling[0], range_downsampling[1])
    num_downsampled_points = int(num_points * downsample_factor)
    if num_downsampled_points < num_points:
        indices = np.random.choice(num_points, num_downsampled_points, replace=False)
        downsampled_points = seg_sample_point[indices]
        downsampled_colors = seg_sample_color[indices]
        downsampled_labels = seg_sample_label[indices]
    else:
        downsampled_points = seg_sample_point
        downsampled_colors = seg_sample_color
        downsampled_labels = seg_sample_label

    return downsampled_points, downsampled_colors, downsampled_labels

def generate_object_segmentation_dataset_wreal_background(objects_dataset_path, target_object_filename, generated_dataset_path, max_num_objects, num_orientations=10, data_config={}, range_downsampling=[0.1, 1.0], max_points_in_box=20480, increase_rate=0.001):
    """
    objects_dataset_path: Path to the directory containing the processed dataset files with specific number of points per object (HDF5 files).
    target_object_filename: The filename of the target object HDF5 file (e.g., "object_1.h5").
    generated_dataset_path: Path to the directory where the generated segmentation dataset will be saved.
    max_num_objects: Maximum number of objects (including the target object) in each segmentation sample.
    num_orientations: Number of different random transformations to generate samples with

    note: the one-hot encoding of the labels is as follows:
        one-hot encoding [1, 0]  (Object of interest)
        one-hot encoding [0, 1]  (Background/clutter)

    """
    print("IN generate_object_segmentation_dataset_wreal_background")

    print("Generated Dataset Path: ", generated_dataset_path)



    # Get file paths
    target_hdf5_file = os.path.join(objects_dataset_path, target_object_filename)
    alien_hdf5_files = sorted(glob.glob(os.path.join(objects_dataset_path, "*.h5")))
    alien_hdf5_files.remove(target_hdf5_file)
    RAW_TEST_SET_DIR = data_config["RAW_TEST_SET_DIR"]
    NUM_SCENES = data_config["NUM_SCENES"]

    # Read the target object data
    with h5py.File(target_hdf5_file, "r") as f:
        target_points = f["point_clouds"][()]  # numpy array
        target_colors = f["color_clouds"][()]  # numpy array
    
    # Read the scene data
    with h5py.File(RAW_TEST_SET_DIR, "r") as f:
        scene_points = f["seg_points"][:]
        scene_colors = f["seg_colors"][:]
        scene_labels = f["seg_labels"][:]
    B_scenes, num_points, _ = scene_points.shape


    num_points_per_object = target_points.shape[1]
    NUM_POINTS_PER_SEG_SAMPLE = num_points_per_object * max_num_objects


    all_seg_sample_points = []
    all_seg_sample_colors = []
    all_seg_sample_labels = []

    for scene_num in range(NUM_SCENES):
        print(f"Processing scene {scene_num} / {NUM_SCENES}")

        # Select Random scene sample
        scene_sample_index = np.random.randint(0, B_scenes)  # Randomly select a scene sample index
        selected_scene_sample_point = scene_points[scene_sample_index, :, :]
        selected_scene_sample_color = scene_colors[scene_sample_index, :, :]
        selected_scene_sample_label = scene_labels[scene_sample_index, :, :]
        downsampled_points, downsampled_colors, downsampled_labels = downsampling(
            selected_scene_sample_point, selected_scene_sample_color, selected_scene_sample_label)
        # Get voxel anywhere in the scene
        downsampled_points, downsampled_colors, downsampled_labels = get_voxel(
            downsampled_points, downsampled_colors, downsampled_labels,
            max_points_in_box=max_points_in_box, increase_rate=increase_rate, centered_in_object=False
        )
        # choose random point in the scene
        rand_point_in_scene_idx = np.random.randint(0, downsampled_points.shape[0])
        rand_point_in_scene = downsampled_points[rand_point_in_scene_idx, :]

        # Select Random target sample
        target_sample_index = np.random.randint(0, target_points.shape[0])  # Randomly select a target sample index
        selected_target_sample_point = target_points[target_sample_index, :, :]
        selected_target_sample_color = target_colors[target_sample_index, :, :]
        selected_target_sample_label = np.concatenate((np.ones((target_points.shape[1], 1)), np.zeros((target_points.shape[1], 1))), axis=1)  # One-hot encoding [1, 0]
        # choose random point in the target sample
        rand_point_in_target_idx = np.random.randint(0, selected_target_sample_point.shape[0])
        rand_point_in_target = selected_target_sample_point[rand_point_in_target_idx, :]
        # center the target sample around the random point in the scene
        selected_target_sample_point = selected_target_sample_point - rand_point_in_target + rand_point_in_scene
        # Build complete scene segmentation sample
        seg_sample_point = downsampled_points
        seg_sample_color = downsampled_colors
        seg_sample_label = downsampled_labels
        seg_sample_point = np.concatenate((seg_sample_point, selected_target_sample_point), axis=0)
        seg_sample_color = np.concatenate((seg_sample_color, selected_target_sample_color), axis=0)
        seg_sample_label = np.concatenate((seg_sample_label, selected_target_sample_label), axis=0)


        # Add Noise Objects
        NUM_ALIEN_OBJECTS = random.randrange(max_num_objects - 1)
        print(f"Processing sample {target_sample_index}, Number of alien objects: {NUM_ALIEN_OBJECTS}")
        # Randomly select alien object files
        copy_alien_files = alien_hdf5_files.copy()
        while len(copy_alien_files) > NUM_ALIEN_OBJECTS:
            copy_alien_files.pop(random.randrange(len(copy_alien_files)))
        # Add alien objects to the segmentation sample
        for i, alien_object_file in enumerate(copy_alien_files):
            with h5py.File(alien_object_file, "r") as f:
                alien_points = f["point_clouds"][()]
                alien_colors = f["color_clouds"][()]

                # Random alien sample selection
                alien_sample_index = random.randrange(alien_points.shape[0])
                selected_alien_sample_point = alien_points[alien_sample_index, :, :]
                selected_alien_sample_color = alien_colors[alien_sample_index, :, :]
                selected_alien_sample_label = np.concatenate((np.zeros((alien_points.shape[1], 1)), np.ones((alien_points.shape[1], 1))), axis=1)  # One-hot encoding [0, 1]
                # choose random point in the scene
                rand_point_in_scene_idx = np.random.randint(0, downsampled_points.shape[0])
                rand_point_in_scene = downsampled_points[rand_point_in_scene_idx, :]
                # choose random point in the alien object
                rand_point_in_alien_idx = np.random.randint(0, selected_alien_sample_point.shape[0])
                rand_point_in_alien = selected_alien_sample_point[rand_point_in_alien_idx, :]
                # center the alien object around the random point in the scene
                selected_alien_sample_point = selected_alien_sample_point - rand_point_in_alien + rand_point_in_scene
                seg_sample_point = np.concatenate((seg_sample_point, selected_alien_sample_point), axis=0)
                seg_sample_color = np.concatenate((seg_sample_color, selected_alien_sample_color), axis=0)
                seg_sample_label = np.concatenate((seg_sample_label, selected_alien_sample_label), axis=0)  # One-hot encoding [0, 1] for alien objects

        downsampled_points, downsampled_colors, downsampled_labels = get_voxel(
            seg_sample_point, seg_sample_color, seg_sample_label,
            max_points_in_box=max_points_in_box, increase_rate=increase_rate, centered_in_object=True
        )
        # Append to the lists
        all_seg_sample_points.append(downsampled_points)
        all_seg_sample_colors.append(downsampled_colors)
        all_seg_sample_labels.append(downsampled_labels)
        if (scene_num+1) % 100 == 0:
            print(f"Processed {(scene_num+1)} scenes so far.")
            # Save the augmented point cloud
            filename = f"REAL_BACKGROUND_{os.path.basename(generated_dataset_path)}_numPoints_{max_points_in_box}_maxObjects_{max_num_objects}_numscenes_{(scene_num+1)}.h5"
            hdf5_filename = os.path.join(os.path.dirname(generated_dataset_path), filename)
            print(f"Saving augmented dataset to: {hdf5_filename}")
            with h5py.File(hdf5_filename, 'w') as f:
                f.create_dataset("seg_points", data=np.asarray(all_seg_sample_points))
                f.create_dataset("seg_colors", data=np.asarray(all_seg_sample_colors))
                f.create_dataset("seg_labels", data=np.asarray(all_seg_sample_labels))

