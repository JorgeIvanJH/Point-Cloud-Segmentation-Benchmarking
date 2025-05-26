import os
import numpy as np
import h5py
import random
import glob
import math
import datetime
import yaml
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
def generate_object_segmentation_dataset(objects_dataset_path, target_object_filename, generated_dataset_path, max_num_objects, num_orientations=10):
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

    all_seg_sample_points = []
    all_seg_sample_colors = []
    all_seg_sample_labels = []

    # Get files
    target_hdf5_file = os.path.join(objects_dataset_path, target_object_filename)
    alien_hdf5_files = sorted(glob.glob(os.path.join(objects_dataset_path, "*.h5")))
    alien_hdf5_files.remove(target_hdf5_file)

    # Read the target object data
    with h5py.File(target_hdf5_file, "r") as f:
        target_points = f["point_clouds"][()]  # numpy array
        target_colors = f["color_clouds"][()]  # numpy array

    num_points_per_object = target_points.shape[1]
    NUM_POINTS_PER_SEG_SAMPLE = num_points_per_object * max_num_objects

    # Iterate through orientations
    for it in range(num_orientations):
        print(f"Orientation Sample {it+1} / {num_orientations}")

        # Process each target sample
        for target_sample_index in range(target_points.shape[0]):
            print(f"Processing target sample {target_sample_index + 1} / {target_points.shape[0]}")

            selected_target_sample_point = target_points[target_sample_index, :, :]
            selected_target_sample_color = target_colors[target_sample_index, :, :]

            # Apply random transformation to target object
            selected_target_sample_point = apply_random_transformation(selected_target_sample_point, num_points_per_object)

            # Prepare segmentation sample (target object)
            seg_sample_point = selected_target_sample_point
            seg_sample_color = selected_target_sample_color
            seg_sample_label = np.concatenate((np.ones((num_points_per_object, 1)), np.zeros((num_points_per_object, 1))), axis=1)  # One-hot encoding [1, 0]

            # Random number of alien objects
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

    # Save the dataset
    hdf5_filename = f"{generated_dataset_path}_segmentation_{timestamp}_numPoints_{num_points_per_object}_maxObjects_{max_num_objects}_numrientations_{num_orientations}.h5"
    with h5py.File(hdf5_filename, 'w') as f:
        f.create_dataset("seg_points", data=np.asarray(all_seg_sample_points))
        f.create_dataset("seg_colors", data=np.asarray(all_seg_sample_colors))
        f.create_dataset("seg_labels", data=np.asarray(all_seg_sample_labels))

    print("Dataset generation complete!")
    return
