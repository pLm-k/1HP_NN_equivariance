import time
from dataclasses import dataclass, field
from math import inf
from typing import Dict
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader
from pathlib import Path

from networks.unet import UNet
import processing.rotation as rt


def setup_visualization_logger(log_file_path: Path = None, console_level: int = logging.INFO):
    """Setup logger for visualization operations."""
    logger = logging.getLogger('visualization')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


class InferenceProcessor:
    """Centralized inference processing with device management and logging."""
    
    def __init__(self, model: UNet, dataloader: DataLoader, device: str, logger: logging.Logger = None):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.norm = dataloader.dataset.dataset.norm
        self.info = dataloader.dataset.dataset.info
        self.model.eval()
        self.logger = logger or logging.getLogger('visualization')
        
        self.logger.info(f"Initialized InferenceProcessor with device: {device}")
        
        # Create rotation processor if needed
        self.rotation_processor = rt.RotationProcessor(model, self.info, device)
    
    def process_single_datapoint(self, x: torch.Tensor, y: torch.Tensor, 
                                rotate_inference: bool = False, mask: bool = False, 
                                crop: bool = False, angle: float = 0) -> tuple:
        """Process single datapoint with consistent device handling and timing."""
        start_time = time.perf_counter()
        
        # Ensure inputs are on correct device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Apply rotation if specified
        if angle != 0:
            x = rt.rotate(x, angle)
            y = rt.rotate(y, angle)
        
        # Prepare input
        x_input = x.unsqueeze(0) if x.dim() == 3 else x
        
        # Apply cropping before inference if needed
        if crop:
            safe_size = rt.get_safe_size(x_input)
            x_input = rt.safe_center_crop(x_input, safe_size)
            y = rt.safe_center_crop(y, safe_size)
        
        # Get inference with detailed timing
        inference_start_time = time.perf_counter()
        if rotate_inference:
            y_out = self.rotation_processor.rotate_and_infer(x, [-1, 0], mask, crop)
            y_out = y_out.unsqueeze(0) if y_out.dim() == 2 else y_out
        else:
            y_out = self.model(x_input)
        
        inference_time = time.perf_counter() - inference_start_time
        
        # Apply masking if needed
        if mask and not crop:
            y = rt.mask_tensor(y)
            y_out = rt.mask_tensor(y_out.squeeze(0)).unsqueeze(0)
        
        total_time = time.perf_counter() - start_time
        
        self.logger.debug(f"Datapoint processed - inference: {inference_time:.4f}s, total: {total_time:.4f}s, shape: {x.shape}")
        
        return x_input, y, y_out, inference_time
    
    def reverse_normalization(self, x: torch.Tensor, y: torch.Tensor, y_out: torch.Tensor):
        """Reverse normalization for plotting."""
        x = self.norm.reverse(x.detach().cpu().squeeze(0), "Inputs")
        y = self.norm.reverse(y.detach().cpu(), "Labels")[0]
        y_out = self.norm.reverse(y_out.detach().cpu().squeeze(0), "Labels")[0]
        return x, y, y_out

@dataclass
class DataToVisualize:
    data: np.ndarray
    name: str
    extent_highs :tuple = (1280,100) # x,y in meters
    imshowargs: Dict = field(default_factory=dict)
    contourfargs: Dict = field(default_factory=dict)
    contourargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        extent = (0,int(self.extent_highs[0]),int(self.extent_highs[1]),0)

        self.imshowargs = {"cmap": "RdBu_r", 
                           "extent": extent}

        self.contourfargs = {"levels": np.arange(10.4, 16, 0.25), 
                             "cmap": "RdBu_r", 
                             "extent": extent}
        
        T_gwf = 10.6
        T_inj_diff = 5.0
        self.contourargs = {"levels" : [np.round(T_gwf + 1, 1)],
                            "cmap" : "Pastel1", 
                            "extent": extent}

        if self.name == "Liquid Pressure [Pa]":
            self.name = "Pressure in [Pa]"
        elif self.name == "Material ID":
            self.name = "Position of the heatpump in [-]"
        elif self.name == "Permeability X [m^2]":
            self.name = "Permeability in [m$^2$]"
        elif self.name == "SDF":
            self.name = "SDF-transformed position in [-]"
    
def visualizations(model: UNet, dataloader: DataLoader, device: str, 
                  amount_datapoints_to_visu: int = inf, plot_path: str = "default", 
                  pic_format: str = "png", rotate_inference: bool = False, 
                  mask: bool = False, crop: bool = False, logger: logging.Logger = None):
    """Generate visualizations with simplified device handling and comprehensive logging."""
    start_time = time.perf_counter()
    logger = logger or logging.getLogger('visualization')
    
    logger.info(f"Starting visualizations - amount: {amount_datapoints_to_visu}, plot_path: {plot_path}")
    logger.info(f"Visualization parameters - rotate_inference: {rotate_inference}, mask: {mask}, crop: {crop}")
    
    print("Visualizing...", end="\r")

    if amount_datapoints_to_visu > len(dataloader.dataset):
        amount_datapoints_to_visu = len(dataloader.dataset)
        logger.warning(f"Requested visualizations ({amount_datapoints_to_visu}) exceed dataset size, using {len(dataloader.dataset)}")

    processor = InferenceProcessor(model, dataloader, device, logger)
    settings_pic = {"format": pic_format, "dpi": 600}

    current_id = 0
    total_inference_time = 0
    
    logger.info(f"Processing {amount_datapoints_to_visu} datapoints for visualization")
    
    for inputs, labels in dataloader:
        for datapoint_id in range(inputs.shape[0]):
            if current_id >= amount_datapoints_to_visu:
                logger.info(f"Completed {current_id} visualizations")
                total_time = time.perf_counter() - start_time
                logger.info(f"Total visualization time: {total_time:.4f}s, avg inference time: {total_inference_time/current_id:.4f}s")
                return

            datapoint_start_time = time.perf_counter()
            name_pic = f"{plot_path}_{current_id}"
            x = inputs[datapoint_id]
            y = labels[datapoint_id]

            # Process datapoint
            x_processed, y_processed, y_out, inference_time = processor.process_single_datapoint(
                x, y, rotate_inference, mask, crop)
            
            total_inference_time += inference_time

            # Reverse normalization for plotting
            x_norm, y_norm, y_out_norm = processor.reverse_normalization(
                x_processed, y_processed, y_out)

            # Calculate loss for this datapoint
            loss = F.mse_loss(y_out.squeeze(), y_processed)
            
            # Prepare and save plot data
            dict_to_plot = prepare_data_to_plot(x_norm, y_norm, y_out_norm, processor.info)
            
            np.save(f'{plot_path}_label_{current_id}.npy', dict_to_plot['t_true'].data.T.numpy())
            np.save(f'{plot_path}_prediction_{current_id}.npy', dict_to_plot['t_out'].data.T.numpy())
            
            plot_datafields(dict_to_plot, name_pic, settings_pic)
            
            datapoint_time = time.perf_counter() - datapoint_start_time
            logger.debug(f"Datapoint {current_id}: loss={loss.item():.6f}, inference_time={inference_time:.4f}s, total_time={datapoint_time:.4f}s")
            
            current_id += 1
            
            # Progress logging every 10 items
            if current_id % 10 == 0:
                logger.info(f"Progress: {current_id}/{amount_datapoints_to_visu} visualizations complete")

    total_time = time.perf_counter() - start_time
    logger.info(f"All {current_id} visualizations completed in {total_time:.4f}s")
    logger.info(f"Average inference time per datapoint: {total_inference_time/current_id:.4f}s")


def infer_all_and_summed_pic(model: UNet, dataloader: DataLoader, device: str, 
                           rotate_inference: bool = False, mask: bool = False, 
                           angle: float = 0, crop: bool = False, logger: logging.Logger = None):
    """Calculate average inference time and summed error with cleaner device handling and comprehensive logging."""
    start_time = time.perf_counter()
    logger = logger or logging.getLogger('visualization')
    
    logger.info(f"Starting infer_all_and_summed_pic - device: {device}, dataset_size: {len(dataloader.dataset)}")
    logger.info(f"Parameters - rotate_inference: {rotate_inference}, mask: {mask}, angle: {angle}, crop: {crop}")
    
    processor = InferenceProcessor(model, dataloader, device, logger)
    
    total_inference_time = 0
    num_samples = 0
    summed_error_pic = None
    total_loss = 0

    for inputs, labels in dataloader:
        for datapoint_id in range(inputs.shape[0]):
            x = inputs[datapoint_id]
            y = labels[datapoint_id]

            # Process datapoint
            _, y_processed, y_out, inference_time = processor.process_single_datapoint(
                x, y, rotate_inference, mask, crop, angle)

            total_inference_time += inference_time
            num_samples += 1
            
            # Calculate loss for this datapoint
            loss = F.mse_loss(y_out.squeeze(), y_processed)
            total_loss += loss.item()

            # Reverse normalization for error calculation
            _, y_norm, y_out_norm = processor.reverse_normalization(
                x.unsqueeze(0), y_processed, y_out)

            # Accumulate error
            error = torch.abs(y_norm - y_out_norm)
            if summed_error_pic is None:
                summed_error_pic = error.cpu()
            else:
                summed_error_pic += error.cpu()
                
            # Progress logging every 100 samples
            if num_samples % 100 == 0:
                logger.info(f"Processed {num_samples} samples - avg_loss: {total_loss/num_samples:.6f}, avg_inference_time: {total_inference_time/num_samples:.4f}s")

    avg_inference_time = total_inference_time / num_samples
    avg_loss = total_loss / num_samples
    avg_error_pic = summed_error_pic / num_samples
    
    # Rotate back if angle was applied
    if angle != 0:
        avg_error_pic = rt.rotate(avg_error_pic.unsqueeze(0), 360 - angle).squeeze(0)
    
    total_time = time.perf_counter() - start_time
    logger.info(f"Inference complete - {num_samples} samples processed in {total_time:.4f}s")
    logger.info(f"Final metrics - avg_loss: {avg_loss:.6f}, avg_inference_time: {avg_inference_time:.4f}s")
    
    return avg_inference_time, avg_error_pic


def infer_all_rotate_and_summed_pic(model: UNet, dataloader: DataLoader, device: str, 
                                  rotate_inference: bool = False, mask: bool = True, 
                                  angle: float = 0, logger: logging.Logger = None):
    """Calculate equivariance error with simplified processing and comprehensive logging."""
    start_time = time.perf_counter()
    logger = logger or logging.getLogger('visualization')
    
    logger.info(f"Starting equivariance analysis - device: {device}, angle: {angle}")
    logger.info(f"Parameters - rotate_inference: {rotate_inference}, mask: {mask}")
    
    processor = InferenceProcessor(model, dataloader, device, logger)
    
    num_samples = 0
    summed_error_pic = None
    total_equivariance_error = 0

    for inputs, _ in dataloader:
        for datapoint_id in range(inputs.shape[0]):
            x = inputs[datapoint_id]
            dummy_y = torch.zeros_like(x[0:1])  # Dummy label for processing

            # Process original datapoint
            _, _, y_out_orig, inference_time_orig = processor.process_single_datapoint(
                x, dummy_y, rotate_inference, mask, False, 0)

            # Process rotated datapoint
            _, _, y_out_rot, inference_time_rot = processor.process_single_datapoint(
                x, dummy_y, rotate_inference, mask, False, angle)

            # Rotate the rotated prediction back
            y_out_rot_back = rt.rotate(y_out_rot, 360 - angle)

            # Reverse normalization
            _, _, y_out_orig_norm = processor.reverse_normalization(
                x.unsqueeze(0), dummy_y, y_out_orig)
            _, _, y_out_rot_norm = processor.reverse_normalization(
                x.unsqueeze(0), dummy_y, y_out_rot_back)

            # Calculate equivariance error
            error = torch.abs(y_out_orig_norm - y_out_rot_norm)
            equivariance_error = error.mean().item()
            total_equivariance_error += equivariance_error
            
            if summed_error_pic is None:
                summed_error_pic = error.cpu()
            else:
                summed_error_pic += error.cpu()

            num_samples += 1
            
            # Progress logging every 50 samples
            if num_samples % 50 == 0:
                avg_equivariance_error = total_equivariance_error / num_samples
                logger.info(f"Processed {num_samples} samples - avg_equivariance_error: {avg_equivariance_error:.6f}")

    avg_equivariance_error = total_equivariance_error / num_samples
    total_time = time.perf_counter() - start_time
    
    logger.info(f"Equivariance analysis complete - {num_samples} samples in {total_time:.4f}s")
    logger.info(f"Final equivariance error: {avg_equivariance_error:.6f}")

    return summed_error_pic / num_samples

def prepare_data_to_plot(x: torch.Tensor, y: torch.Tensor, y_out: torch.Tensor, info: dict):
    """Prepare data for plotting with consistent formatting."""
    temp_max = max(y.max(), y_out.max())
    temp_min = min(y.min(), y_out.min())
    extent_highs = (np.array(info["CellsSize"][:2]) * x.shape[-2:])

    dict_to_plot = {
        "t_true": DataToVisualize(y, "Label: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "t_out": DataToVisualize(y_out, "Prediction: Temperature in [°C]", extent_highs, {"vmax": temp_max, "vmin": temp_min}),
        "error": DataToVisualize(torch.abs(y-y_out), "Absolute error in [°C]", extent_highs),
    }
    
    inputs = info["Inputs"].keys()
    for input_name in inputs:
        index = info["Inputs"][input_name]["index"]
        dict_to_plot[input_name] = DataToVisualize(x[index], input_name, extent_highs)

    return dict_to_plot


def plot_datafields(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict):
    """Plot datafields with consistent formatting."""
    fontsize = 8
    num_subplots = len(data)
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)
    
    for index, (name, datapoint) in enumerate(data.items()):
        plt.sca(axes[index])
        plt.title(datapoint.name, fontsize=fontsize, pad=10)
        plt.imshow(datapoint.data.T, **datapoint.imshowargs)
        plt.gca().invert_yaxis()
        plt.ylabel("x [m]", fontsize=fontsize)
        plt.tick_params(axis='both', labelsize=fontsize)
        _aligned_colorbar(fontsize=fontsize)
        plt.tick_params(axis='both', labelsize=fontsize)

    plt.sca(axes[-1])
    plt.xlabel("y [m]", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{name_pic}.{settings_pic['format']}", **settings_pic)
    plt.close()


def plot_avg_error_cellwise(dataloader, summed_error_pic, settings_pic: dict):
    """Plot average error cellwise."""
    info = dataloader.dataset.dataset.info
    extent_highs = (np.array(info["CellsSize"][:2]) * dataloader.dataset[0][0][0].shape)
    extent = (0, int(extent_highs[0]), int(extent_highs[1]), 0)

    plt.figure()
    plt.imshow(summed_error_pic.T, cmap="RdBu_r", extent=extent)
    plt.gca().invert_yaxis()
    plt.ylabel("x [m]")
    plt.xlabel("y [m]")
    plt.title("Cellwise averaged error [°C]")
    _aligned_colorbar()
    plt.tight_layout()
    plt.savefig(f"{settings_pic['folder']}/avg_error.{settings_pic['format']}", format=settings_pic['format'])
    plt.close()


def plot_avg_error_rotated_cellwise(dataloader, summed_error_pic_dif, settings_pic: dict, angle: float):
    """Plot average error cellwise between predictions of rotated and unrotated inputs."""
    info = dataloader.dataset.dataset.info
    extent_highs = (np.array(info["CellsSize"][:2]) * dataloader.dataset[0][0][0].shape)
    extent = (0, int(extent_highs[0]), int(extent_highs[1]), 0)

    plt.figure()
    plt.imshow(summed_error_pic_dif.T, cmap="RdBu_r", extent=extent)
    plt.gca().invert_yaxis()
    plt.ylabel("x [m]")
    plt.xlabel("y [m]")
    plt.title("Difference cellwise averaged error [°C]")
    _aligned_colorbar()
    plt.tight_layout()
    plt.savefig(f"{settings_pic['folder']}/avg_error_dif_{angle}.{settings_pic['format']}", format=settings_pic['format'])
    plt.close()


def _aligned_colorbar(fontsize: int = -1, *args, **kwargs):
    """Create aligned colorbar."""
    cax = make_axes_locatable(plt.gca()).append_axes("right", size=0.15, pad=0.05)
    if fontsize > 0:
        offset_text = cax.yaxis.get_offset_text()
        offset_text.set_fontsize(fontsize)
    plt.colorbar(*args, cax=cax, **kwargs)


def run_comprehensive_visualization_with_logging(model: UNet, dataloader: DataLoader, device: str, 
                                                output_folder: Path, amount_datapoints: int = 10):
    """
    Comprehensive visualization function with full logging for SLURM jobs.
    
    This function demonstrates all visualization capabilities with detailed logging
    suitable for monitoring SLURM job progress and collecting metrics.
    
    Args:
        model: Trained UNet model
        dataloader: DataLoader with test data
        device: Device for inference ('cuda' or 'cpu')
        output_folder: Path to folder for saving outputs and logs
        amount_datapoints: Number of datapoints to visualize
    
    Returns:
        dict: Summary metrics including average loss and inference times
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_folder / "visualization_log.txt"
    logger = setup_visualization_logger(log_file)
    
    logger.info("="*60)
    logger.info("STARTING COMPREHENSIVE VISUALIZATION ANALYSIS")
    logger.info("="*60)
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Device: {device}")
    logger.info(f"Dataset size: {len(dataloader.dataset)}")
    logger.info(f"Amount to visualize: {amount_datapoints}")
    
    total_start_time = time.perf_counter()
    
    # 1. Generate individual visualizations
    logger.info("\n" + "-"*40)
    logger.info("PHASE 1: Individual Visualizations")
    logger.info("-"*40)
    
    viz_start_time = time.perf_counter()
    visualizations(
        model=model,
        dataloader=dataloader,
        device=device,
        amount_datapoints_to_visu=amount_datapoints,
        plot_path=str(output_folder / "individual"),
        pic_format="png",
        logger=logger
    )
    viz_time = time.perf_counter() - viz_start_time
    logger.info(f"Individual visualizations completed in {viz_time:.4f}s")
    
    # 2. Calculate overall inference metrics
    logger.info("\n" + "-"*40)
    logger.info("PHASE 2: Overall Inference Analysis")
    logger.info("-"*40)
    
    inference_start_time = time.perf_counter()
    avg_inference_time, avg_error_pic = infer_all_and_summed_pic(
        model=model,
        dataloader=dataloader,
        device=device,
        logger=logger
    )
    inference_analysis_time = time.perf_counter() - inference_start_time
    
    logger.info(f"Inference analysis completed in {inference_analysis_time:.4f}s")
    
    # 3. Equivariance analysis (if applicable)
    logger.info("\n" + "-"*40)
    logger.info("PHASE 3: Equivariance Analysis")
    logger.info("-"*40)
    
    equivariance_start_time = time.perf_counter()
    equivariance_error_pic = infer_all_rotate_and_summed_pic(
        model=model,
        dataloader=dataloader,
        device=device,
        angle=90,  # Test with 90-degree rotation
        logger=logger
    )
    equivariance_time = time.perf_counter() - equivariance_start_time
    
    logger.info(f"Equivariance analysis completed in {equivariance_time:.4f}s")
    
    # Calculate final metrics
    total_time = time.perf_counter() - total_start_time
    
    # Summary metrics
    metrics = {
        'avg_inference_time': avg_inference_time,
        'total_visualization_time': viz_time,
        'total_inference_analysis_time': inference_analysis_time,
        'total_equivariance_time': equivariance_time,
        'total_time': total_time,
        'datapoints_processed': amount_datapoints,
        'dataset_size': len(dataloader.dataset)
    }
    
    # Log final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Total processing time: {total_time:.4f}s")
    logger.info(f"Average inference time per sample: {avg_inference_time:.4f}s")
    logger.info(f"Individual visualizations time: {viz_time:.4f}s")
    logger.info(f"Inference analysis time: {inference_analysis_time:.4f}s")
    logger.info(f"Equivariance analysis time: {equivariance_time:.4f}s")
    logger.info(f"Datapoints processed: {amount_datapoints}")
    logger.info(f"Total dataset size: {len(dataloader.dataset)}")
    
    # Save metrics to file for SLURM job collection
    metrics_file = output_folder / "metrics_summary.txt"
    with open(metrics_file, 'w') as f:
        f.write("VISUALIZATION METRICS SUMMARY\n")
        f.write("="*40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Metrics summary saved to: {metrics_file}")
    logger.info("COMPREHENSIVE VISUALIZATION ANALYSIS COMPLETE")
    logger.info("="*60)
    
    return metrics
