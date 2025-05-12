model = "PointNet"

import numpy as np

import h5py
import yaml
from data.data_provider import SegmentationDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics.classification import MulticlassMatthewsCorrCoef

import matplotlib.pyplot as plt

from models.item_pointnet2_torch import PointNet2_SegHead
from models.item_pointnet2_torch import PointNet2_SegLoss


with open("config.yaml", "r") as f:
    c = yaml.safe_load(f)
    try:
        MODE = "TRAIN"
        DATA_PATH = c["DATA_PATH"]
        OBJECT_NAME = c["OBJECT_NAME"]
        MODELS_PATH = c["MODELS_PATH"]
        NUM_POINTS_PER_SEG_SAMPLE = c["NUM_POINTS_PER_SEG_SAMPLE"]
        DEVICE = c["DEVICE"]
        BATCH_SIZE = c[MODE]["BATCH_SIZE"]
        LR = c[MODE]["LR"]
        EPOCHS = c[MODE]["EPOCHS"]
        if DEVICE == "cuda":
            if torch.cuda.is_available():
                DEVICE = torch.device("cuda")
            else:
                print("CUDA is not available. Using CPU instead.")
                DEVICE = torch.device("cpu")
    
    except KeyError as e:
        print(f"Key {e} not found in config.yaml. Please check the file.")
        raise

dataset = SegmentationDataset(DATA_PATH, OBJECT_NAME)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size) # 80% training set
val_size = dataset_size - train_size # 20% validation set
train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)


points, targets = next(iter(train_dataloader))
NUM_CLASSES = targets.unique().shape[0]
seg_model = PointNet2_SegHead(num_points=NUM_POINTS_PER_SEG_SAMPLE, m=NUM_CLASSES)


alpha = np.ones(NUM_CLASSES)
gamma = 1
optimizer = optim.Adam(seg_model.parameters(), lr=LR)
num_iterations_in_epoch = len(train_dataloader)
step_size = int(num_iterations_in_epoch * 8)     # as recommended by the paper \ref{smith2017cyclical}
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR, max_lr=0.01, step_size_up=step_size, cycle_momentum=False) # cycle_momentum=False to be compatible with Adam
criterion = PointNet2_SegLoss(alpha=alpha, gamma=gamma, dice=True).to(DEVICE)
seg_model = seg_model.to(DEVICE)
mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)


# TRAINING LOOP AND VALIDATION LOOP SAVE