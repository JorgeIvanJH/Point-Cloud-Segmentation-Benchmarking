import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import random
import glob
import numpy as np
import os



class SegmentationDataset(Dataset):
    # number_of_alien_objects_to_train_against must be even number
    def __init__(self, 
                 object_segmentation_dataset_directory, 
                 target_object_dataset_name):
        self.dataset_dir = object_segmentation_dataset_directory
        self.target_obj = target_object_dataset_name
        
        # Get the target object segmentation file
        #target_hdf5_file = self.dataset_dir + target_object_name + "_segmentation_" + str(self.num_points_per_seg_sample)
        #target_hdf5_file = self.dataset_dir + "coffee_nescafe_3in1_original_6cups_1200_2048_segmentation_20480_12000"
        #target_hdf5_file = self.dataset_dir + "coffee_nescafe_3in1_original_6cups_1200_2048_segmentation_20480_4800"
        #target_hdf5_file = self.dataset_dir + "shampoo_head_and_shoulders_citrus_400ml_1200_2048_segmentation_4800"
        # target_hdf5_file = self.dataset_dir + target_object_dataset_name
        target_hdf5_file = os.path.join(self.dataset_dir, target_object_dataset_name)
        # Target class training data collection
        with h5py.File(target_hdf5_file, "r") as f:
            point_data = f["seg_points"][()]  # returns as a numpy array
            color_data = f["seg_colors"][()]  # returns as a numpy array
            label_data = f["seg_labels"][()]  # returns as a numpy array
        self.dataset_samples = np.concatenate( (point_data, color_data),axis=2)
        self.dataset_labels = label_data
        #print(self.dataset_samples.shape)
        #print(self.dataset_labels.shape)
        
    def __len__(self):
        return len(self.dataset_labels)

    def __getitem__(self, index):
        point_cloud_sample = self.dataset_samples[index]
        label = self.dataset_labels[index]
        return torch.from_numpy(point_cloud_sample).float(), torch.from_numpy(label).long()


def get_data_loaders(object_segmentation_dataset_directory, 
                     target_object_dataset_name,shuffle=True,train_size=0.8,batch_size=32, seed=42):
    seed_gen = torch.Generator().manual_seed(seed)
    dataset = SegmentationDataset(object_segmentation_dataset_directory, target_object_dataset_name)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[train_size, (1-train_size)], generator=seed_gen)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, valid_dataloader
