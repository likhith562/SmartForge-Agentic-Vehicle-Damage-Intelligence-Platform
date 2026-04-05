"""
SmartForge — MiDaS Depth Estimation
=====================================
Wraps the MiDaS monocular depth-estimation model from the isl-org/MiDaS
TorchHub repository.  Provides two pure functions used by perception_node:

    run_midas_depth(image_rgb)                    → np.ndarray (depth map)
    compute_deformation_index(depth_map, mask)    → float

The deformation index converts MiDaS 2D→3D depth reasoning into a scalar
that quantifies physical deformation within a detected damage region:

    deformation_index = var(depth[mask]) / (global_var(depth) + ε)

A score near 0 means the masked region is as flat as the surrounding
vehicle surface (normal, undamaged panel).
A higher score means the depth variance inside the mask is elevated
compared to the global scene variance — indicating a physical dent,
crumple, or structural deformation.

This value flows into compute_severity() and the false_positive_gate
(FLAT_SURFACE check) downstream.

Usage
-----
    from src.cv.depth import run_midas_depth, compute_deformation_index

    depth_map = run_midas_depth(image_rgb)
    rv        = compute_deformation_index(depth_map, sam_mask)
"""

from typing import Optional

import numpy as np
import torch


def run_midas_depth(image_rgb: np.ndarray) -> np.ndarray:
    """
    Run MiDaS small (MiDaS_small) on an RGB image and return a depth map
    resized to match the input spatial dimensions.

    Uses GPU when available, falls back to CPU automatically.

    Parameters
    ----------
    image_rgb : np.ndarray — RGB image of shape (H, W, 3), dtype uint8

    Returns
    -------
    np.ndarray — depth map of shape (H, W), float32
                 Values are relative (not metric), suitable only for
                 comparing regions within the same image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and transforms from TorchHub (cached after first download)
    midas = torch.hub.load(
        "isl-org/MiDaS", "MiDaS_small", verbose=False
    )
    midas.to(device).eval()

    transforms = torch.hub.load(
        "isl-org/MiDaS", "transforms", verbose=False
    )
    transform = transforms.small_transform

    # Run inference
    input_tensor = transform(image_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size           = image_rgb.shape[:2],
            mode           = "bicubic",
            align_corners  = False,
        ).squeeze()

    depth_map: np.ndarray = prediction.cpu().numpy()
    return depth_map


def compute_deformation_index(
    depth_map: np.ndarray,
    mask: Optional[np.ndarray],
) -> float:
    """
    Compute the relative deformation index for a masked region.

    Formula
    -------
        global_var = var(depth_map) + ε
        mask_var   = var(depth_map[mask == True])
        index      = mask_var / global_var

    Interpretation
    --------------
    ≈ 0.0          → flat surface (normal undamaged panel)
    0.001 – 0.005  → minor surface variation (scratch / paint chip)
    0.005 – 0.02   → moderate deformation (medium dent)
    > 0.02         → significant structural deformation (severe dent / crumple)

    Parameters
    ----------
    depth_map : np.ndarray — full-image MiDaS depth output (H, W)
    mask      : np.ndarray or None — SAM binary mask (H, W), dtype bool
                When None, returns 0.0.

    Returns
    -------
    float — relative deformation index (rounded to 6 decimal places)
    """
    if mask is None:
        return 0.0

    global_var = float(np.var(depth_map)) + 1e-6

    # Extract depth values within the mask
    depth_in_mask = depth_map[mask == True]

    if len(depth_in_mask) == 0:
        return 0.0

    mask_var = float(np.var(depth_in_mask))
    index    = mask_var / global_var

    return round(index, 6)
