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

from data.data_provider import get_data_loaders

from arg_extractor import get_config

c = get_config()
print(c)

train_dataloader, valid_dataloader = get_data_loaders(
    object_segmentation_dataset_directory=c["DATA_PATH"],
    target_object_dataset_name=c["OBJECT_NAME"],
    shuffle=c["SHUFFLE"],
    train_size=c["TRAIN_SIZE"],
    batch_size=c["BATCH_SIZE"],
    seed=c["SEED"]
)


# points, targets = next(iter(train_dataloader))
# NUM_CLASSES = targets.unique().shape[0]
# seg_model = PointNet2_SegHead(num_points=NUM_POINTS_PER_SEG_SAMPLE, m=NUM_CLASSES)


# alpha = np.ones(NUM_CLASSES)
# gamma = 1
# optimizer = optim.Adam(seg_model.parameters(), lr=LR)
# num_iterations_in_epoch = len(train_dataloader)
# step_size = int(num_iterations_in_epoch * 8)     # as recommended by the paper \ref{smith2017cyclical}
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR, max_lr=0.01, step_size_up=step_size, cycle_momentum=False) # cycle_momentum=False to be compatible with Adam
# criterion = PointNet2_SegLoss(alpha=alpha, gamma=gamma, dice=True).to(DEVICE)
# seg_model = seg_model.to(DEVICE)
# mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)


# # TRAINING LOOP AND VALIDATION LOOP SAVE