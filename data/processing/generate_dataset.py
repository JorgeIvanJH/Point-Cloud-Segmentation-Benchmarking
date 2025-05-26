import open3d as o3d
import os
import glob
import numpy as np
import h5py
from matplotlib import pyplot as plt
from generate_hdf5_dataset_with_padding import generate_hdf5_dataset_with_padding
from segmentation_dataset_prep import generate_object_segmentation_dataset
import random
import getpass
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml

CONFIG_PATH = "../../config.yaml" # Make sure path os correct and has the required parameters


def load_yaml_config(filepath="config.yaml"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found at {filepath}")
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
    
def processing_of_individual_objects(yaml_config, data_config, POINTCLOUD_SAVE_DIR):

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

def gen_object_segmentation_dataset(yaml_config, data_config, POINTCLOUD_SAVE_DIR):
    TARGET_OBJECT_NAME = data_config["TARGET_OBJECT_NAME"]
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
    print("Saving segmentation dataset to: ", SEGMENTATION_SAVE_DIR)
    generate_object_segmentation_dataset( POINTCLOUD_SAVE_DIR, my_product_pcdataset_hdf5_file, SEGMENTATION_SAVE_DIR, data_config["MAX_NUM_OBJECTS"], data_config["NUM_ORIENTATIONS"] )

if __name__ == '__main__':
    yaml_config = load_yaml_config(CONFIG_PATH)
    data_config = yaml_config["DATA"]
    POINTCLOUD_SAVE_DIR = os.path.join(data_config["POINTCLOUD_SAVE_DIR"], str(data_config["NUM_POINTS_PER_OBJECT"]))
    if not os.path.exists(POINTCLOUD_SAVE_DIR): # Create folder for dataset with NUM_POINTS_PER_OBJECT 
        os.makedirs(POINTCLOUD_SAVE_DIR)

    # UNCOMMENT LATER
    # print("Generating processed HDF5 dataset of individual objects")
    # processing_of_individual_objects(yaml_config, data_config, POINTCLOUD_SAVE_DIR)
    
    print("Generating processed HDF5 dataset of object segmentation")
    gen_object_segmentation_dataset(yaml_config, data_config, POINTCLOUD_SAVE_DIR)

    print("Dataset generation complete. Processed point clouds saved to:", POINTCLOUD_SAVE_DIR)


