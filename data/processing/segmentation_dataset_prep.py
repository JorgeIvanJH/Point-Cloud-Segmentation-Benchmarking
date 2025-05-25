import os
import numpy as np
import h5py
import random
#import time
import glob
import math
import getpass

import yaml

# TODO: Set seed

CONFIG_PATH = "../../config.yaml"



# We want that:
# 1. Each sample has fixed number of points NUM_POINTS
# 2. Each sample has one target object (for the moment till we figure out how to detect multiple instances!)
# 3. Each sample has variable number of alien objects (from 0 till 9)
# 4. Each object in the sample should be located in a different pose with respect to each other object, all objects should move specially the target object


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





def generate_object_segmentation_dataset( objects_dataset_path, target_object_filename, generated_dataset_path, max_num_objects , num_orientations=10):
    """
    objects_dataset_path: Path to the directory containing the processed dataset files with specific number of points per object (HDF5 files).
    target_object_filename: The filename of the target object HDF5 file (e.g., "object_1.h5").
    generated_dataset_path: Path to the directory where the generated segmentation dataset will be saved.
    max_num_objects: Maximum number of objects (including the target object) in each segmentation sample.

    note: the one-hot encoding of the labels is as follows:
        one-hot encoding [1, 0]  (Object of interest)
        one-hot encoding [0, 1]  (Background/clutter)

    """
    print("generated_dataset_path: ", generated_dataset_path)
    all_seg_sample_points=[]
    all_seg_sample_colors=[]
    all_seg_sample_labels=[]
    
    # Get files
    target_hdf5_file = os.path.join(objects_dataset_path, target_object_filename)
    alien_hdf5_files = sorted(glob.glob(os.path.join(objects_dataset_path, "*.h5")))
    alien_hdf5_files.remove(target_hdf5_file)
    
    with h5py.File(target_hdf5_file, "r") as f:
        target_points = f["point_clouds"][()]  # returns as a numpy array
        target_colors = f["color_clouds"][()]  # returns as a numpy array
 
    num_points_per_objects = target_points.shape[1] # assuming all objects will have the same sample size
    NUM_POINTS_PER_SEG_SAMPLE = num_points_per_objects * max_num_objects  # Total number of points in each segmentation sample

    for it in range(num_orientations):
        num_individual_samples = target_points.shape[0]//num_orientations
        for j in range(int(num_individual_samples)):
            target_sample_index = random.randint(0, target_points.shape[0] - 1)

            selected_target_sample_point = target_points[target_sample_index,:,:]
            selected_target_sample_color = target_colors[target_sample_index,:,:]
   
            # Convert to homogeneous coordinates (add a column of ones to the 3d points to be able to multiply with 4x4 homogeneous transform afterwards)
            selected_target_sample_point = np.concatenate( (selected_target_sample_point, np.ones((num_points_per_objects,1))),axis=1)

            # Apply homogeneous transformation
            random_homogeneous_transformation = random_homogeneous_3d_translation(-0.25,0.25,-0.25,0.25,0.0,0.25) * random_homogeneous_3d_rotation(-180,180,-180,180,-180,180)
            
            selected_target_sample_point = np.matmul( random_homogeneous_transformation, np.transpose(selected_target_sample_point) )
            selected_target_sample_point = np.transpose(selected_target_sample_point)
            
            # Remove the column of ones added previously
            selected_target_sample_point = np.delete( selected_target_sample_point, 3, 1 )
            
            # Add the target object sample to the collective segmentation sample
            seg_sample_point = selected_target_sample_point
            seg_sample_color = selected_target_sample_color
            seg_sample_label = np.concatenate( (np.ones((num_points_per_objects,1)), np.zeros((num_points_per_objects,1))),axis=1)  #one-hot encoding [1, 0]  (Object of interest)

            

            # Randomly generate the number of alien objects in this sample
            NUM_ALIEN_OBJECTS = random.randrange(max_num_objects-1)
            print("iteration: ", str(it+1) + " , processing sample: " + str(target_sample_index) + " , number of alien objects = " + str(NUM_ALIEN_OBJECTS))

            # Remove files randomly to keep only the required number of alien objects NUM_ALIEN_OBJECTS
            copy_alien_files = alien_hdf5_files.copy()
            while len(copy_alien_files) > NUM_ALIEN_OBJECTS:
                copy_alien_files.pop(random.randrange(len(copy_alien_files)))
                    
            
            # Alien objects data collection
            for i, alien_object_file in enumerate(copy_alien_files):
                with h5py.File(alien_object_file, "r") as f:
                    alien_points = f["point_clouds"][()]
                    alien_colors = f["color_clouds"][()]
                    
                    # Pick a random sample of the alien object i
                    alien_sample_index = random.randrange(alien_points.shape[0])
                    selected_alien_sample_point = alien_points[alien_sample_index,:,:]
                    selected_alien_sample_color = alien_colors[alien_sample_index,:,:]

                    # Convert to homogeneous coordinates (add a column of ones to the 3d points to be able to multiply with 4x4 homogeneous transform afterwards)
                    selected_alien_sample_point = np.concatenate( (selected_alien_sample_point, np.ones((num_points_per_objects,1))),axis=1)

                    # Apply homogeneous transformation
                    random_homogeneous_transformation = random_homogeneous_3d_translation(-0.25,0.25,-0.25,0.25,0.0,0.25) * random_homogeneous_3d_rotation(-180,180,-180,180,-180,180)
                    selected_alien_sample_point = np.matmul( random_homogeneous_transformation, np.transpose(selected_alien_sample_point) )
                    selected_alien_sample_point = np.transpose(selected_alien_sample_point)

                    # Remove the column of ones added previously
                    selected_alien_sample_point = np.delete( selected_alien_sample_point, 3, 1 )

                    # Add the alien object sample to the collective segmentation sample
                    seg_sample_point = np.concatenate( (seg_sample_point, selected_alien_sample_point), axis=0 )
                    seg_sample_color = np.concatenate( (seg_sample_color, selected_alien_sample_color), axis=0 )
                    seg_sample_label = np.concatenate( (seg_sample_label, np.concatenate( (np.zeros((num_points_per_objects,1)), np.ones((num_points_per_objects,1))),axis=1)), axis=0 )  # one-hot encoding [0, 1]  (Background/clutter)


            # Pad the remaining sample size with zeros since we have varying number of objects per segmentation sample generated
            seg_sample_point = np.concatenate( (seg_sample_point, np.zeros((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_point.shape[0], 3)) ), axis=0 )
            seg_sample_color = np.concatenate( (seg_sample_color, np.zeros((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_color.shape[0], 3)) ), axis=0 )
            seg_sample_label = np.concatenate( (seg_sample_label, np.concatenate( (np.zeros(((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_label.shape[0]),1)), np.ones(((NUM_POINTS_PER_SEG_SAMPLE - seg_sample_label.shape[0]),1))),axis=1)), axis=0 )  # one-hot encoding [0, 1]  (Background/clutter)
        
            # add current segmentation sample to the dataset
            all_seg_sample_points.append(seg_sample_point)
            all_seg_sample_colors.append(seg_sample_color)
            all_seg_sample_labels.append(seg_sample_label)
    

    # Save the segmentation samples to a new file
    hdf5_filename = generated_dataset_path + "_segmentation_" + str(NUM_POINTS_PER_SEG_SAMPLE) + "_" + str(num_orientations*num_individual_samples) + ".h5"

    with h5py.File(hdf5_filename, 'w') as f:
        # Create a point clouds dataset in the file
        f.create_dataset("seg_points", data = np.asarray(all_seg_sample_points))
        f.create_dataset("seg_colors", data = np.asarray(all_seg_sample_colors))
        f.create_dataset("seg_labels", data = np.asarray(all_seg_sample_labels))
    print("done!")
    return




