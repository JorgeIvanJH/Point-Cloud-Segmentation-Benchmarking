import open3d as o3d
import os
import glob
import numpy as np
import h5py
from matplotlib import pyplot as plt
from single_object_preparation import generate_object_segmentation_dataset,generate_object_segmentation_dataset_wreal_background
import random
import getpass
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml

CONFIG_PATH = "../../config.yaml" # Make sure path os correct and has the required parameters


import open3d as o3d
import os
import glob
import numpy as np
import h5py
import random
import json


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


def load_yaml_config(filepath="config.yaml"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found at {filepath}")
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
    
def processing_of_individual_objects(yaml_config, data_config, POINTCLOUD_SAVE_DIR):

    list_of_products = [f for f in os.listdir(POINTCLOUD_SAVE_DIR) if f.endswith('.h5')]
    if len(list_of_products) != data_config["NUM_PRODUCTS_IN_DATASET"]:
        
        print("Num prods: ", len(list_of_products))
        print("number of points per object: ", data_config["NUM_POINTS_PER_OBJECT"])
        print("saving processed objects to: ", POINTCLOUD_SAVE_DIR)
        for product_name in os.listdir(data_config["RAW_DATASET_DIR"]):
            print("Processing product: ", product_name)
            MY_PRODUCT_READ_DIR = os.path.join(data_config["RAW_DATASET_DIR"], product_name)
            NUM_SAMPLES = len(os.listdir(MY_PRODUCT_READ_DIR)) # Number of samples is the number of .pcd files in the product directory, each a different sample of the same object with variations in pose, lighting, etc.
            my_product_pcdataset_name = product_name + "_" + str(NUM_SAMPLES) + "_" + str(data_config["NUM_POINTS_PER_OBJECT"])
            my_product_pcdataset_hdf5_file = my_product_pcdataset_name + ".h5"
            MY_PRODUCT_PCDATASET_SAVE_DIR = os.path.join(POINTCLOUD_SAVE_DIR, my_product_pcdataset_hdf5_file)
            generate_hdf5_dataset_with_padding(MY_PRODUCT_READ_DIR, MY_PRODUCT_PCDATASET_SAVE_DIR, data_config["NUM_POINTS_PER_OBJECT"], seed=yaml_config["SEED"])
    print("HDF5 Files of all products ready.")

def gen_object_segmentation_dataset(data_config, POINTCLOUD_SAVE_DIR, WITH_REAL_BACKGROUND=False):
    TARGET_OBJECT_NAME = data_config["TARGET_OBJECT_NAME"] # e.g: "ketchup_heinz_400ml"
    list_of_products = os.listdir(POINTCLOUD_SAVE_DIR)
    prod_matching_name = [prod for prod in list_of_products if TARGET_OBJECT_NAME in prod] # Filter products that match the object name
    if len(prod_matching_name) != 1:
        raise ValueError(f"Expected exactly one product matching '{TARGET_OBJECT_NAME}', found {len(prod_matching_name)}. Please check the dataset directory.")
    my_product_pcdataset_hdf5_file = prod_matching_name[0] 
    print("Generating segmentation dataset for product: ", my_product_pcdataset_hdf5_file)
    SEGMENTATION_SAVE_DIR = os.path.join(POINTCLOUD_SAVE_DIR,"segmentation_dataset")
    if not os.path.exists(SEGMENTATION_SAVE_DIR): # Create folder for segmentation dataset
        os.makedirs(SEGMENTATION_SAVE_DIR)
    SEGMENTATION_SAVE_DIR = os.path.join(SEGMENTATION_SAVE_DIR, TARGET_OBJECT_NAME)
    if not WITH_REAL_BACKGROUND:
        print("Saving segmentation dataset to: ", SEGMENTATION_SAVE_DIR)
        generate_object_segmentation_dataset( POINTCLOUD_SAVE_DIR, my_product_pcdataset_hdf5_file, SEGMENTATION_SAVE_DIR, data_config["MAX_NUM_OBJECTS"], data_config["NUM_ORIENTATIONS"] )
    else:
        print("Generating segmentation dataset with real background. This may take a while...")
        generate_object_segmentation_dataset_wreal_background( POINTCLOUD_SAVE_DIR, my_product_pcdataset_hdf5_file, SEGMENTATION_SAVE_DIR, data_config["MAX_NUM_OBJECTS"], data_config["NUM_ORIENTATIONS"], data_config )


if __name__ == '__main__':
    yaml_config = load_yaml_config(CONFIG_PATH)
    data_config = yaml_config["DATA"]
    WITH_REAL_BACKGROUND = data_config["WITH_REAL_BACKGROUND"] # If True, the background will be taken from real scenes in the test set
    POINTCLOUD_SAVE_DIR = os.path.join(data_config["POINTCLOUD_SAVE_DIR"], str(data_config["NUM_POINTS_PER_OBJECT"]))
    print("POINTCLOUD_SAVE_DIR: ", POINTCLOUD_SAVE_DIR)
    if not os.path.exists(POINTCLOUD_SAVE_DIR): # Create folder for dataset with NUM_POINTS_PER_OBJECT 
        os.makedirs(POINTCLOUD_SAVE_DIR)
    print("Generating processed HDF5 dataset of individual objects")
    processing_of_individual_objects(yaml_config, data_config, POINTCLOUD_SAVE_DIR)
    print("Generating processed HDF5 dataset of object segmentation")
    gen_object_segmentation_dataset(data_config, POINTCLOUD_SAVE_DIR, WITH_REAL_BACKGROUND)
    print("Dataset generation complete. Processed point clouds saved to:", POINTCLOUD_SAVE_DIR)


