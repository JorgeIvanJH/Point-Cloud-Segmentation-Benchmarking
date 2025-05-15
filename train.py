import os
import numpy as np
import torch
import torch.optim as optim
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from experiment_builder import ExperimentBuilder
from data.data_provider import get_data_loaders
from models.selector import select_model
from arg_extractor import get_config

def main():
    c = get_config()
    print(c)
    DEVICE = torch.device("cuda" if (torch.cuda.is_available() and c["DEVICE"] == "cuda" ) else "cpu")

    # Data Providers
    train_dataloader, valid_dataloader, test_dataloader = get_data_loaders(
        train_val_dir=c["DATA_PATH"],
        train_val_name=c["OBJECT_NAME"],
        test_dir=c["DATA_PATH"], # TODO: change to test data
        test_name=c["OBJECT_NAME"], # TODO: change to test data
        batch_size=c["BATCH_SIZE"],
        train_size=c["TRAIN_SIZE"],
        shuffle=c["SHUFFLE"],
        seed=c["SEED"],
        use_colors=c["USE_COLORS"],
        normalize=c["NORMALIZE"],
        num_workers=c["NUM_WORKERS"],
        pin_memory=c["PIN_MEMORY"],

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
                                        test_data=test_dataloader,
                                        device=DEVICE,
                                        continue_from_epoch=c["CONTINUE_FROM_EPOCH"],
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        loss_criterion =criterion,
                                        metrics=metrics
                                        )  # build an experiment object
    experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
    print(f"Experiment metrics: {experiment_metrics}")
    print(f"Test metrics: {test_metrics}")

if __name__ == '__main__':

  # optional but recommended for Windows
    main()