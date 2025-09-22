import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from itertools import product
from typing import Optional


class RotationProcessor:
    """Centralized rotation processing with consistent device handling."""
    
    def __init__(self, model: torch.nn.Module, info: dict, device: str):
        self.model = model
        self.info = info
        self.device = device
        
    def rotate(self, data: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate tensor counter-clockwise by given angle."""
        return TF.rotate(data, angle, interpolation=InterpolationMode.BILINEAR)
    
    def rotate_and_infer(self, datapoint: torch.Tensor, grad_vec: list, 
                        mask: bool = False, crop: bool = False) -> torch.Tensor:
        """Rotate datapoint to align with grad_vec, infer, and rotate back."""
        # Ensure input is on correct device
        datapoint = datapoint.to(self.device)
        
        # Calculate gradient and get angle for aligning data point
        angle = get_rotation_angle(get_pressure_grad(datapoint, self.info), grad_vec)
        x = self.rotate(datapoint, angle)

        if crop:
            safe_size = get_safe_size(x)
            x = safe_center_crop(x, safe_size)
        elif mask:
            x = mask_tensor(x)
        
        # Ensure x is on correct device for model inference
        x = x.to(self.device)
        
        # Get inference
        with torch.no_grad():
            y_out = self.model(x.unsqueeze(0))

        # Rotate result back
        y_out = self.rotate(y_out, 360 - angle)
        return y_out.squeeze(0)

    def rotate_and_infer_batch(self, batch: torch.Tensor, grad_vec: list,
                              mask: bool = False, crop: bool = False) -> torch.Tensor:
        """Process entire batch with rotation."""
        batch = batch.to(self.device)
        y_out_list = []
        
        for datapoint in batch:
            result = self.rotate_and_infer(datapoint, grad_vec, mask, crop)
            y_out_list.append(result)
        
        return torch.stack(y_out_list)


# Standalone rotation function
def rotate(data: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate tensor counter-clockwise by given angle."""
    return TF.rotate(data, angle, interpolation=InterpolationMode.BILINEAR)

# get angle to rotate a counter-clockwise to match b's direction
def get_rotation_angle(a: list, b: list) -> float:
    """Calculate rotation angle to align vector a with vector b."""
    # calculate the dot product and the determinant
    dot_product = np.dot(a, b)
    determinant = a[0] * b[1] - a[1] * b[0]
    
    # calculate the angle
    angle = np.degrees(np.arctan2(determinant, dot_product))
    
    # turn angle positive if necessary
    if angle < 0:
        angle += 360
    
    return angle

# rotate tensors with gradient correction for vector fields
def rotate_w_gradient(data: torch.Tensor, angle: float, info) -> torch.Tensor:
    """Rotate tensor with gradient correction for vector fields.
    
    This function handles rotation of tensors that contain both scalar and vector fields.
    Vector fields (like gradients) need special handling during rotation.
    """
    if isinstance(info, str):
        # If info is a path, load the info
        import yaml
        with open(info, 'r') as f:
            info = yaml.safe_load(f)
    
    rotated_data = rotate(data, angle)
    
    # Check if we have gradient fields that need special rotation handling
    if isinstance(info, dict) and 'Inputs' in info:
        # Look for gradient fields in the info structure
        for field_name, field_info in info['Inputs'].items():
            if 'Gradient' in field_name and 'index' in field_info:
                idx = field_info['index']
                if idx < data.shape[0]:
                    # For gradient fields, we need to rotate the vector components
                    # This is a simplified version - actual implementation depends on 
                    # how gradients are stored in your tensors
                    grad_x = rotated_data[idx]
                    if idx + 1 < data.shape[0]:
                        grad_y = rotated_data[idx + 1]
                        # Rotate gradient vector
                        angle_rad = np.radians(angle)
                        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                        new_grad_x = cos_a * grad_x - sin_a * grad_y
                        new_grad_y = sin_a * grad_x + cos_a * grad_y
                        rotated_data[idx] = new_grad_x
                        rotated_data[idx + 1] = new_grad_y
    
    return rotated_data


# get pressure gradient encoded by the data points pressure field
def get_pressure_grad(datapoint: torch.Tensor, info: dict) -> list:
    """Extract pressure gradient from datapoint."""
    # get indices for calculating gradient and leave border of size 5 so that masks dont interfere
    p_ind = info['Inputs']['Liquid Pressure [Pa]']['index']
    center = int(datapoint[p_ind].shape[0]/2)
    start = 5
    end = datapoint[p_ind].shape[0] - 5
    dif = end - start

    # calculate gradient
    return [(datapoint[p_ind][end][center].item() - datapoint[p_ind][start][center].item())/dif, 
            (datapoint[p_ind][center][end].item() - datapoint[p_ind][center][start].item())/dif]


def get_safe_size(tensor: torch.Tensor, depth: int = 3) -> int:
    """Calculate safe size for rotation without corners."""
    H, W = tensor.shape[-2:]
    assert H == W, "Tensor must be square"
    
    safe_size = int(H / np.sqrt(2))
    safe_size -= safe_size % 2 ** depth
    return safe_size

def safe_center_crop(tensors: torch.Tensor, safe_size: int) -> torch.Tensor:
    """Crop square tensor to safe size that fits any rotated version."""
    start = (tensors.shape[-2] - safe_size) // 2
    end = start + safe_size
    
    return tensors[..., start:end, start:end]

def build_mask(s: int, dim: int = 2, dtype=torch.float32, device: Optional[str] = None) -> torch.Tensor:
    """Build circular mask tensor."""
    mask = torch.zeros(1, 1, *[s] * dim, dtype=dtype, device=device)
    c = (s-1) / 2  # center of the tensor
    r_max = c**2  # maximum radius squared for the circle to fit

    for k in product(range(s), repeat=dim):
        r = sum((x - c)**2 for x in k)
        if r <= r_max:
            mask[(..., *k)] = 1.  # inside the circle
        else:
            mask[(..., *k)] = 0.  # outside the circle
    return mask

def mask_size(s: int, dim: int = 2) -> int:
    """Get number of cells inside circular mask."""
    c = (s-1) / 2  # center of the tensor
    r_max = c**2  # maximum radius squared for the circle to fit
    pixels = 0

    for k in product(range(s), repeat=dim):
        r = sum((x - c)**2 for x in k)
        if r <= r_max:
            pixels += 1  # inside the circle
    return pixels

def mask_tensor(data: torch.Tensor) -> torch.Tensor:
    """Apply circular mask to tensor."""
    data_out = torch.zeros_like(data)
    mask = build_mask(data.shape[-1], dtype=data.dtype, device=data.device)
    
    # Apply mask to each channel
    for i in range(data.shape[0]):
        data_out[i] = data[i] * mask.squeeze()
    
    return data_out

def mask_batch(batch: torch.Tensor) -> torch.Tensor:
    """Apply circular mask to batch of tensors."""
    return torch.stack([mask_tensor(data) for data in batch])
