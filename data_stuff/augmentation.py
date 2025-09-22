"""
Data augmentation utilities for simulation datasets.
Separated from dataset classes for better organization and reusability.
"""

from typing import List, Tuple
import numpy as np
import random

import torch
from torch.utils.data import Subset

from data_stuff.dataset import TrainDataset
from processing.rotation import (
    mask_tensor, rotate, get_rotation_angle, get_pressure_grad, 
    safe_center_crop, get_safe_size
)


class DataAugmentation:
    """Centralized data augmentation utilities."""
    
    @staticmethod
    def augment_data(dataset, augmentation_n: int = 0, mask: bool = False, 
                    angle: int = 0, crop: bool = False) -> Subset:
        """
        Augment data by adding rotated data points to the original dataset.

        Args:
            dataset: Original dataset to augment
            augmentation_n: Number of augmented data points per original point. 
                          Setting to -1 adds rotations of 90°, 180°, and 270°.
            mask: Apply circular mask to data fields if True
            angle: Rotate all data points by this angle before augmenting
            crop: Apply safe cropping after augmentation
        
        Returns:
            Subset: Augmented dataset
        """
        np.random.seed(42)
        
        # Determine base rotation angles
        if angle == -1:
            angles = np.random.randint(0, 360, len(dataset)).tolist()
        else:
            angles = [angle] * len(dataset)
        
        # Prepare base data
        base_data = DataAugmentation._prepare_base_data(dataset, angles, mask)
        augmented_dataset = TrainDataset(dataset.dataset.path)
        
        # Add original (potentially rotated) data
        for input_tensor, label_tensor, run_id in base_data:
            augmented_dataset.add_item(input_tensor, label_tensor, run_id)
        
        # Add augmented variations
        DataAugmentation._add_augmented_variations(
            augmented_dataset, base_data, augmentation_n, mask)
        
        if crop:
            augmented_dataset = DataAugmentation.crop_data(
                Subset(augmented_dataset, list(range(len(augmented_dataset)))))
        
        return Subset(augmented_dataset, list(range(len(augmented_dataset))))
    
    @staticmethod
    def _prepare_base_data(dataset, angles: List[int], mask: bool) -> List[Tuple]:
        """Prepare base data with initial rotations and optional masking."""
        base_data = []
        
        for i in range(len(dataset)):
            input_tensor = rotate(dataset[i][0], angles[i])
            label_tensor = rotate(dataset[i][1], angles[i])
            run_id = dataset.dataset.get_run_id(i)
            
            if mask:
                input_tensor = mask_tensor(input_tensor)
                label_tensor = mask_tensor(label_tensor)
            
            base_data.append((input_tensor, label_tensor, run_id))
        
        return base_data
    
    @staticmethod
    def _add_augmented_variations(augmented_dataset: TrainDataset, base_data: List[Tuple], 
                                 augmentation_n: int, mask: bool):
        """Add augmented variations to the dataset."""
        for input_tensor, label_tensor, run_id in base_data:
            if augmentation_n > 0:
                # Add random rotations
                for _ in range(augmentation_n):
                    rot_angle = np.random.rand() * 360
                    DataAugmentation._add_rotated_item(
                        augmented_dataset, input_tensor, label_tensor, 
                        run_id, rot_angle, mask)
            
            elif augmentation_n < 0:
                # Add fixed 90° rotations
                for rot_angle in [90, 180, 270]:
                    DataAugmentation._add_rotated_item(
                        augmented_dataset, input_tensor, label_tensor, 
                        run_id, rot_angle, mask)
    
    @staticmethod
    def _add_rotated_item(dataset: TrainDataset, input_tensor: torch.Tensor, 
                         label_tensor: torch.Tensor, run_id: str, 
                         rot_angle: float, mask: bool):
        """Add a single rotated item to the dataset."""
        rotated_input = rotate(input_tensor, rot_angle)
        rotated_label = rotate(label_tensor, rot_angle)
        new_run_id = f"{run_id}_rot_{rot_angle:.1f}"
        
        if mask:
            rotated_input = mask_tensor(rotated_input)
            rotated_label = mask_tensor(rotated_label)
        
        dataset.add_item(rotated_input, rotated_label, new_run_id)
    
    @staticmethod
    def restrict_data(dataset, data_n: int = -1, seed: int = 1) -> Subset:
        """Restrict dataset to data_n points."""
        if data_n <= 0 or data_n >= len(dataset):
            return dataset
        
        random.seed(seed)
        
        restricted_dataset = TrainDataset(dataset.dataset.path)
        data_points = random.sample([
            (dataset[i][0], dataset[i][1], dataset.dataset.get_run_id(i)) 
            for i in range(len(dataset))
        ], data_n)

        for input_tensor, label_tensor, run_id in data_points:
            restricted_dataset.add_item(input_tensor, label_tensor, run_id)
        
        print(f"Dataset restricted to {len(restricted_dataset)} samples")
        return Subset(restricted_dataset, list(range(len(restricted_dataset))))

    @staticmethod
    def remove_angles(dataset_in, remove_ranges: List[Tuple[float, float]] = None) -> Subset:
        """Remove data points within specified angle ranges."""
        if not remove_ranges:
            return dataset_in
        
        dataset = dataset_in.dataset
        filtered_dataset = TrainDataset(dataset.path)
        
        for i in range(len(dataset)):
            input_tensor, label_tensor = dataset[i]
            angle = get_rotation_angle(
                get_pressure_grad(input_tensor, dataset.info), [-1, 0])
            
            # Check if angle is NOT in any remove range
            if not any(start <= angle < end for start, end in remove_ranges):
                filtered_dataset.add_item(
                    input_tensor, label_tensor, dataset.get_run_id(i))

        return Subset(filtered_dataset, list(range(len(filtered_dataset))))

    @staticmethod
    def crop_data(dataset_in, depth: int = 3) -> Subset:
        """Crop dataset to safe size that avoids blank corners after rotation."""
        dataset = dataset_in.dataset
        cropped_dataset = TrainDataset(dataset.path)
        
        # Get safe size from first label
        first_label = dataset[0][1]
        safe_size = get_safe_size(first_label, depth)

        # Crop all items
        for i in range(len(dataset)):
            input_tensor, label_tensor = dataset[i]
            cropped_dataset.add_item(
                safe_center_crop(input_tensor, safe_size),
                safe_center_crop(label_tensor, safe_size),
                dataset.get_run_id(i)
            )

        return Subset(cropped_dataset, list(range(len(cropped_dataset))))

    @staticmethod
    def rotate_data(dataset, grad_vec: List[float] = None):
        """Rotate data points to align with specified direction."""
        if grad_vec is None:
            grad_vec = [-1, 0]
        
        # Normalize gradient vector
        grad_vec = np.array(grad_vec)
        grad_vec = grad_vec / np.linalg.norm(grad_vec)
        
        rotated_dataset = TrainDataset(dataset.path)
        alignment_errors = []
        
        for i in range(len(dataset)):
            input_tensor, label_tensor = dataset[i]
            
            # Calculate rotation angle to align with grad_vec
            current_grad = get_pressure_grad(input_tensor, dataset.info)
            angle = get_rotation_angle(current_grad, grad_vec.tolist())
            
            # Apply rotation
            rotated_input = rotate(input_tensor, angle)
            rotated_label = rotate(label_tensor, angle)
            rotated_dataset.add_item(
                rotated_input, rotated_label, dataset.get_run_id(i))
            
            # Calculate alignment error
            rotated_grad = np.array(get_pressure_grad(rotated_input, dataset.info))
            rotated_grad = rotated_grad / np.linalg.norm(rotated_grad)
            error = np.abs(grad_vec - rotated_grad)
            alignment_errors.append(error)
        
        mean_error = np.mean(alignment_errors, axis=0)
        print(f'Mean alignment error: {mean_error}')

        return rotated_dataset