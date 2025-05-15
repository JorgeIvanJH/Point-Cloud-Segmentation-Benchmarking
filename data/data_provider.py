import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import os

class SegmentationDataset(Dataset):
    def __init__(self, 
                 dataset_dir, 
                 dataset_name, 
                 use_colors=True, 
                 normalize=True):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.use_colors = use_colors
        self.normalize = normalize

        target_hdf5_file = os.path.join(self.dataset_dir, self.dataset_name)

        if not os.path.exists(target_hdf5_file):
            raise FileNotFoundError(f"HDF5 file not found at: {target_hdf5_file}")

        with h5py.File(target_hdf5_file, "r") as f:
            required_keys = ["seg_points", "seg_colors", "seg_labels"]
            for key in required_keys:
                if key not in f:
                    raise KeyError(f"Key '{key}' not found in HDF5 file.")

            point_data = f["seg_points"][()]  # [N, P, 3]
            color_data = f["seg_colors"][()]  # [N, P, 3]
            label_data = f["seg_labels"][()]  # [N, P, C] or [N, P]

        if self.use_colors:
            self.dataset_samples = np.concatenate((point_data, color_data), axis=2)  # [N, P, 6]
        else:
            self.dataset_samples = point_data  # [N, P, 3]

        self.dataset_labels = label_data  # [N, P, C] or [N, P]

    def __len__(self):
        return len(self.dataset_labels)

    def __getitem__(self, index):
        point_cloud_sample = self.dataset_samples[index]
        label = self.dataset_labels[index]

        if self.normalize:
            point_cloud_sample[:, :3] = self._normalize_pointcloud(point_cloud_sample[:, :3])

        return torch.from_numpy(point_cloud_sample).float(), torch.from_numpy(label).long()

    def _normalize_pointcloud(self, pc):
        pc = pc - np.mean(pc, axis=0)
        scale = np.max(np.linalg.norm(pc, axis=1))
        return pc / scale

    def __repr__(self):
        return (f"SegmentationDataset(samples={len(self)}, "
                f"target='{self.dataset_name}', "
                f"use_colors={self.use_colors}, "
                f"normalize={self.normalize})")


def get_data_loaders(train_val_dir,
                     train_val_name,
                     test_dir,
                     test_name,
                     batch_size=32,
                     train_size=0.8,
                     shuffle=True,
                     seed=42,
                     use_colors=True,
                     normalize=True,
                     num_workers=4,
                     pin_memory=True):

    # Load the train/validation dataset
    full_dataset = SegmentationDataset(train_val_dir, train_val_name,
                                       use_colors=use_colors, normalize=normalize)

    # Split into train/val
    seed_gen = torch.Generator().manual_seed(seed)
    total_len = len(full_dataset)
    train_len = int(train_size * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len], generator=seed_gen)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    
    # Load the test dataset
    test_dataset = SegmentationDataset(test_dir, test_name,
                                       use_colors=use_colors, normalize=normalize)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
