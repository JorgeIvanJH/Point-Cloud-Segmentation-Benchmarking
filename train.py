import os
import numpy as np
import torch
import torch.optim as optim
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from experiment_builder import ExperimentBuilder
from data.data_provider import get_data_loaders
from models.selector import select_model
from arg_extractor import get_config

c = get_config()
print(c)
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and c["DEVICE"] == "cuda" ) else "cpu")

# Data Providers
train_dataloader, valid_dataloader = get_data_loaders(
    object_segmentation_dataset_directory=c["DATA_PATH"],
    target_object_dataset_name=c["OBJECT_NAME"],
    shuffle=c["SHUFFLE"],
    train_size=c["TRAIN_SIZE"],
    batch_size=c["BATCH_SIZE"],
    seed=c["SEED"]
)

# Model
points, targets = next(iter(train_dataloader))
NUM_CLASSES = targets.unique().shape[0]
seg_model, criterion = select_model(c["MODEL_NAME"], c, DEVICE, NUM_CLASSES)


# Training parameters and performance metrics
optimizer = optim.Adam(seg_model.parameters(), lr=c["LR"], weight_decay=c["WEIGHT_DECAY"])
num_iterations_in_epoch = len(train_dataloader)
step_size = int(num_iterations_in_epoch * 8)     # as recommended by the paper \ref{smith2017cyclical}
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=c["LR"], max_lr=0.01, step_size_up=step_size, cycle_momentum=False)
mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)
def compute_iou(predictions,targets):
    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)
    intersection = torch.sum(predictions == targets) # true positives
    union = len(predictions) + len(targets) - intersection
    return intersection / union
metrics = {} # every metric should receive (preds, target) only, in that order
metrics["iou"] = compute_iou
metrics["mcc"] = mcc_metric


# TRAINING LOOP AND VALIDATION LOOP SAVE
conv_experiment = ExperimentBuilder(model=seg_model,
                                    experiment_name=c["EXPERIMENT_NAME"] + "_" + c["MODEL_NAME"],
                                    num_epochs=c["EPOCHS"],
                                    train_data=train_dataloader, 
                                    val_data=valid_dataloader,
                                    test_data=valid_dataloader, # TODO: change to test data
                                    device=DEVICE,
                                    continue_from_epoch=c["CONTINUE_FROM_EPOCH"],
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    loss_criterion =criterion,
                                    metrics=metrics
                                    )  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
