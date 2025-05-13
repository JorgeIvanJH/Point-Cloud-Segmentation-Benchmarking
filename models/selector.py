import numpy as np
import torch


def pointnet(c, device, num_classes):
    print(f"Using PointNet model for segmentation on {c['OBJECT_NAME']}")
    
    from models.pointnet import PointNetSegHead, PointNetSegLoss
    
    model = PointNetSegHead(
        num_points=c["NUM_POINTS_PER_SEG_SAMPLE"],
        m=num_classes
    ).to(device)

    loss_fn = PointNetSegLoss(
        alpha=np.ones(num_classes),
        gamma=1,
        dice=True
    ).to(device)

    return model, loss_fn


def pointnet2(c, device, num_classes):
    print(f"Using PointNet2 model for segmentation on {c['OBJECT_NAME']}")

    from models.pointnet2 import PointNet2_SegHead, PointNet2_SegLoss

    model = PointNet2_SegHead(
        num_points=c["NUM_POINTS_PER_SEG_SAMPLE"],
        m=num_classes
    ).to(device)

    loss_fn = PointNet2_SegLoss(
        alpha=np.ones(num_classes),
        gamma=1,
        dice=True
    ).to(device)

    return model, loss_fn


def select_model(model_name: str, config: dict, device: torch.device, num_classes: int):
    """Select and initialize the appropriate segmentation model."""

    model_registry = {
        "pointnet": pointnet,
        "pointnet2": pointnet2
    }

    model_fn = model_registry.get(model_name.lower())
    if model_fn is None:
        raise ValueError(f"[ERROR] Model '{model_name}' not found. Available: {list(model_registry.keys())}")

    model, loss_fn = model_fn(config, device, num_classes)
    return model, loss_fn

