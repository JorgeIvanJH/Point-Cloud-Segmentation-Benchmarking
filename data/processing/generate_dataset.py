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
    
def processing_of_individual_objects():
    
    yaml_config = load_yaml_config(CONFIG_PATH)
    data_config = yaml_config["DATA"]
    POINTCLOUD_SAVE_DIR = os.path.join(data_config["POINTCLOUD_SAVE_DIR"], str(data_config["NUM_POINTS_PER_OBJECT"]))
    if not os.path.exists(POINTCLOUD_SAVE_DIR): # Create folder for dataset with NUM_POINTS_PER_OBJECT 
        os.makedirs(POINTCLOUD_SAVE_DIR)
    print("number of points per object: ", data_config["NUM_POINTS_PER_OBJECT"])
    print("saving processed objects to: ", POINTCLOUD_SAVE_DIR)
    for product_name in os.listdir(data_config["RAW_DATASET_DIR"]):
        print("Processing product: ", product_name)
        MY_PRODUCT_READ_DIR = os.path.join(data_config["RAW_DATASET_DIR"], product_name)
        NUM_SAMPLES = len(os.listdir(MY_PRODUCT_READ_DIR)) # Number of samples is the number of .pcd files in the product directory, each a different sample of the same object with variations in pose, lighting, etc.
        my_product_pcdataset_name = product_name + "_" + str(NUM_SAMPLES) + "_" + str(data_config["NUM_POINTS_PER_OBJECT"])
        my_product_pcdataset_hdf5_file = my_product_pcdataset_name + ".h5"
        MY_PRODUCT_PCDATASET_SAVE_DIR = os.path.join(POINTCLOUD_SAVE_DIR, my_product_pcdataset_hdf5_file)
        generate_hdf5_dataset_with_padding(MY_PRODUCT_READ_DIR, MY_PRODUCT_PCDATASET_SAVE_DIR, data_config["NUM_POINTS_PER_OBJECT"])

def main():
    print("Generating processed HDF5 dataset of individual objects")
    processing_of_individual_objects()

if __name__ == '__main__':
    main()
    



