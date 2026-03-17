import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from itertools import product, repeat


# function to rotate one datapoint counter-clockwise (with pressure as input)
def rotate(data: torch.tensor, angle: int) -> torch.tensor:
    return TF.rotate(data, angle, interpolation=InterpolationMode.BILINEAR)


# rotate a datapoint such that direction matches specified direction and return rerotated prediction (with pressure as input)
def rotate_and_infer(
    datapoint: torch.tensor,
    grad_vec: list,
    model: torch.nn.Module,
    info,
    device: str,
    crop: bool = False,
    return_angle: bool = False,
) -> torch.tensor:
    # calculate gradient and get angle for aligning data point
    angle = get_rotation_angle(get_pressure_grad(datapoint, info), grad_vec)
    x = rotate(datapoint, angle)

    if crop:
        safe_size = get_safe_size(x.cpu())
        x = safe_center_crop(x.cpu(), safe_size).to(device)

    # get inference
    y_out = model(x.unsqueeze(0))

    if return_angle:
        return y_out, angle

    # rotate result back
    y_out = rotate(y_out, 360 - angle)
    return y_out


# rotate a batch such that direction matches specified direction and return rerotated inference (with pressure as input)
def rotate_and_infer_batch(
    batch: torch.tensor,
    grad_vec: list,
    model: torch.nn.Module,
    info,
    device: str,
    crop: bool = False,
    return_angles: bool = False,
) -> torch.tensor:
    y_out_list = []
    angles = []
    for datapoint in batch:
        if return_angles:
            y_out, angle = rotate_and_infer(
                datapoint, grad_vec, model, info, device, crop, return_angle=True
            )
            y_out_list.append(y_out.squeeze(0))
            angles.append(angle)
        else:
            y_out_list.append(
                rotate_and_infer(
                    datapoint, grad_vec, model, info, device, crop
                ).squeeze(0)
            )

    if return_angles:
        return torch.stack(y_out_list), angles
    return torch.stack(y_out_list)


# get angle to rotate a counter-clockwise to match b's direction
def get_rotation_angle(a: list, b: list) -> int:
    # calculate the dot product and the determinant
    dot_product = np.dot(a, b)
    determinant = a[0] * b[1] - a[1] * b[0]

    # calculate the angle
    angle = np.degrees(np.arctan2(determinant, dot_product))

    # turn angle positive if necessary
    if angle < 0:
        angle += 360

    return angle


# get pressure gradient encoded by the data points pressure field
def get_pressure_grad(datapoint: torch.tensor, info) -> list:
    # get indices for calculating gradient
    p_ind = info["Inputs"]["Liquid Pressure [Pa]"]["index"]
    center = int(datapoint[p_ind].shape[0] / 2)
    start = 5
    end = datapoint[p_ind].shape[0] - 5
    dif = end - start

    # calculate gradient
    return [
        (datapoint[p_ind][end][center].item() - datapoint[p_ind][start][center].item())
        / dif,
        (datapoint[p_ind][center][end].item() - datapoint[p_ind][center][start].item())
        / dif,
    ]


def get_safe_size(tensor, depth=3):
    H, W = tensor.shape[-2:]
    assert H == W, "Tensor must be square"

    safe_size = int(H / np.sqrt(2))
    safe_size -= safe_size % 2**depth
    return safe_size


def safe_center_crop(tensors, safe_size):
    """
    Crop square tensor batch (C,H,W) to maximal safe size
    that fits any rotated version without blank corners.

    Returns: cropped tensor batch
    """
    start = (tensors.shape[-2] - safe_size) // 2
    end = start + safe_size

    return tensors[..., start:end, start:end]
