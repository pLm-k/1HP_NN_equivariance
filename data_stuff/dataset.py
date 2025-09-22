import os
import pathlib
from typing import List, Tuple

import torch
import yaml
from torch.utils.data import Dataset

from data_stuff.transforms import NormalizeTransform


class BaseDataset(Dataset):
    """Base class for all simulation datasets with common functionality."""
    
    def __init__(self, path: str):
        super().__init__()
        self.path = pathlib.Path(path)
        self.info = self._load_info()
        self.norm = NormalizeTransform(self.info)
    
    def _load_info(self) -> dict:
        """Load dataset info from YAML file."""
        with open(self.path / "info.yaml", "r") as f:
            return yaml.safe_load(f)
    
    @property
    def input_channels(self) -> int:
        return len(self.info["Inputs"])

    @property
    def output_channels(self) -> int:
        return len(self.info["Labels"])


class SimulationDataset(BaseDataset):
    """Standard simulation dataset for loading input/label pairs."""
    
    def __init__(self, path: str):
        super().__init__(path)
        self.input_names = sorted(os.listdir(self.path / "Inputs"))
        self.label_names = sorted(os.listdir(self.path / "Labels"))
        
        if len(self.input_names) != len(self.label_names):
            raise ValueError("Number of inputs and labels does not match!")

    def __len__(self) -> int:
        return len(self.input_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = torch.load(self.path / "Inputs" / self.input_names[index])
        label_tensor = torch.load(self.path / "Labels" / self.label_names[index])
        return input_tensor, label_tensor
    
    def get_run_id(self, index: int) -> str:
        return self.input_names[index]


class TrainDataset(BaseDataset):
    """Dataset for training with support for dynamic data addition."""
    
    def __init__(self, path: str):
        super().__init__(path)
        self.inputs: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.run_ids: List[str] = []

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.labels[index]
    
    def add_item(self, input_tensor: torch.Tensor, label_tensor: torch.Tensor, run_id: str):
        """Add a single item to the dataset."""
        self.inputs.append(input_tensor)
        self.labels.append(label_tensor)
        self.run_ids.append(run_id)

    def get_run_id(self, index: int) -> str:
        return self.run_ids[index]
class DatasetExtend1(BaseDataset):
    """Dataset for extend plumes problem - first stage."""
    
    def __init__(self, path: str, box_size: int = 64):
        super().__init__(path)
        self.input_names = sorted(os.listdir(self.path / "Inputs"))
        self.label_names = sorted(os.listdir(self.path / "Labels"))
        self.spatial_size = torch.load(self.path / "Inputs" / self.input_names[0]).shape[1:]
        self.box_size = box_size

    def __len__(self) -> int:
        return len(self.input_names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = torch.load(self.path / "Inputs" / self.input_names[idx])[:, :self.box_size, :]
        label_tensor = torch.load(self.path / "Labels" / self.label_names[idx])[:, :self.box_size, :]
        return input_tensor, label_tensor


class DatasetExtend2(BaseDataset):
    """Dataset for extend plumes problem - second stage."""
    
    def __init__(self, path: str, skip_per_dir: int = 4, box_size: int = 64):
        super().__init__(path)
        self.input_names = sorted(os.listdir(self.path / "Inputs"))
        self.label_names = sorted(os.listdir(self.path / "Labels"))
        self.spatial_size = torch.load(self.path / "Inputs" / self.input_names[0]).shape[1:]
        self.box_size = box_size
        self.skip_per_dir = skip_per_dir
        self.dp_per_run = ((self.spatial_size[0]) // self.box_size - 2) * (self.box_size // self.skip_per_dir)
        print(f"dp_per_run: {self.dp_per_run}, spatial_size: {self.spatial_size}, "
              f"box_size: {self.box_size}, skip_per_dir: {self.skip_per_dir}")

    @property
    def input_channels(self) -> int:
        return len(self.info["Inputs"]) + 1  # +1 for temperature

    def __len__(self) -> int:
        return len(self.input_names) * self.dp_per_run

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        run_id, box_id = self._idx_to_pos(idx)
        
        start_pos = box_id * self.skip_per_dir
        input_slice = slice(start_pos + self.box_size, start_pos + 2 * self.box_size)
        temp_slice = slice(start_pos, start_pos + self.box_size)
        
        input_tensor = torch.load(self.path / "Inputs" / self.input_names[run_id])[:, input_slice, :]
        input_T = torch.load(self.path / "Labels" / self.input_names[run_id])[:, temp_slice, :]
        
        assert input_tensor.shape[1:] == input_T.shape[1:], \
            f"Shapes do not match: {input_tensor.shape} vs {input_T.shape}"
        
        input_combined = torch.cat((input_tensor, input_T), dim=0)
        label_tensor = torch.load(self.path / "Labels" / self.label_names[run_id])[:, input_slice, :]
        
        return input_combined, label_tensor

    def _idx_to_pos(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to (run_id, box_id)."""
        return idx // self.dp_per_run, idx % self.dp_per_run + 1


# Utility functions
def get_splits(n: int, splits: List[float]) -> List[int]:
    """Convert fractional splits to integer splits that sum to n."""
    splits = [int(n * s) for s in splits[:-1]]
    splits.append(n - sum(splits))
    return splits