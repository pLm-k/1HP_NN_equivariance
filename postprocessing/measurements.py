import torch
from torch.nn import MSELoss, L1Loss
from torch.utils.data import DataLoader
import time
import yaml
import logging
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import processing.rotation as rt

from networks.unet import UNet
from processing.solver import Solver
from data_stuff.utils import SettingsTraining


def setup_measurement_logger(log_file_path: Path = None, console_level: int = logging.INFO):
    """Setup logger for measurement operations."""
    logger = logging.getLogger('measurements')
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


class MeasurementProcessor:
    """Centralized measurement processing with device management and logging."""
    
    def __init__(self, model: UNet, settings: SettingsTraining, logger: logging.Logger = None):
        self.model = model
        self.device = settings.device
        self.settings = settings
        self.model.eval()
        self.logger = logger or logging.getLogger('measurements')
        
        self.logger.info(f"Initialized MeasurementProcessor with device: {self.device}")
        
    def get_dataset_info(self, dataloaders: Dict[str, DataLoader]):
        """Extract dataset info consistently."""
        train_loader = dataloaders["train"]
        if self.settings.problem == "allin1":
            norm = train_loader.dataset.norm
            output_channels = train_loader.dataset.output_channels
            info = train_loader.dataset.info
        else:  # 2stages, 1hp
            norm = train_loader.dataset.dataset.norm
            output_channels = train_loader.dataset.dataset.output_channels
            info = train_loader.dataset.dataset.info
        
        self.logger.debug(f"Dataset info extracted - output_channels: {output_channels}")
        return norm, output_channels, info
    
    def process_batch(self, x: torch.Tensor, y: torch.Tensor, info: dict,
                     rotate_inference: bool = False, mask: bool = False, 
                     crop: bool = False) -> tuple:
        """Process a batch with consistent device handling and timing."""
        batch_start_time = time.perf_counter()
        
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Apply cropping if needed
        if crop:
            safe_size = rt.get_safe_size(y)
            y = rt.safe_center_crop(y, safe_size)
        
        # Get predictions with timing
        inference_start_time = time.perf_counter()
        if rotate_inference:
            rotation_processor = rt.RotationProcessor(self.model, info, self.device)
            y_pred_list = []
            for i in range(x.shape[0]):
                pred = rotation_processor.rotate_and_infer(x[i], [-1, 0], mask, crop)
                y_pred_list.append(pred)
            y_pred = torch.stack(y_pred_list)
        else:
            if crop:
                safe_size = rt.get_safe_size(x)
                x = rt.safe_center_crop(x, safe_size)
            y_pred = self.model(x)
        
        inference_time = time.perf_counter() - inference_start_time
        
        # Apply masking if needed
        if mask and not crop:
            y = rt.mask_batch(y)
            y_pred = rt.mask_batch(y_pred)
        
        # Ensure same spatial dimensions
        if y.shape[2:] != y_pred.shape[2:]:
            required_size = y_pred.shape[2:]
            start_pos = ((y.shape[2] - required_size[0])//2, (y.shape[3] - required_size[1])//2)
            y = y[:, :, start_pos[0]:start_pos[0]+required_size[0], 
                  start_pos[1]:start_pos[1]+required_size[1]]
        
        batch_time = time.perf_counter() - batch_start_time
        
        self.logger.debug(f"Batch processed - size: {x.shape}, inference_time: {inference_time:.4f}s, total_time: {batch_time:.4f}s")
        
        return y, y_pred, inference_time

def measure_loss(model: UNet, dataloaders: Dict[str, DataLoader], settings: SettingsTraining, 
                vT_case: str = "temperature", rotate_inference: bool = False, 
                mask: bool = False, crop: bool = False, log_file_path: Path = None):
    """Measure losses with comprehensive logging and timing."""
    
    # Setup logger
    if log_file_path is None and hasattr(settings, 'destination') and settings.destination:
        log_file_path = Path(settings.destination) / "measurements.log"
    
    logger = setup_measurement_logger(log_file_path)
    logger.info("=" * 60)
    logger.info("STARTING LOSS MEASUREMENT")
    logger.info("=" * 60)
    logger.info(f"Settings: rotate_inference={rotate_inference}, mask={mask}, crop={crop}")
    logger.info(f"Variable type: {vT_case}")
    
    processor = MeasurementProcessor(model, settings, logger)
    norm, output_channels, info = processor.get_dataset_info(dataloaders)
    
    if vT_case == "temperature":
        pbt_threshold = [0.1]  # [°C] only relevant for temperature
        logger.info(f"Temperature threshold for PBT: {pbt_threshold}")
    
    results = {}
    total_start_time = time.perf_counter()

    for case, dataloader in dataloaders.items():
        case_start_time = time.perf_counter()
        logger.info(f"\nProcessing dataset: {case} ({len(dataloader.dataset)} samples)")
        
        mse_loss = torch.zeros(output_channels)
        mae_closs = torch.zeros(output_channels)
        rmse_closs = torch.zeros(output_channels)
        if vT_case == "temperature":
            pbt_closs = torch.zeros(output_channels)

        total_inference_time = 0.0
        batch_count = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            # Process batch with timing
            y_processed, y_pred, inference_time = processor.process_batch(x, y, info, rotate_inference, mask, crop)
            total_inference_time += inference_time
            batch_count += 1

            # Log progress every 10 batches or for small datasets
            if batch_idx % max(1, len(dataloader) // 10) == 0:
                logger.debug(f"Batch {batch_idx+1}/{len(dataloader)} - inference: {inference_time:.4f}s")

            # Normalized losses
            for channel in range(y_pred.shape[1]):
                mse_loss[channel] += MSELoss(reduction="sum")(y_pred[:, channel], y_processed[:, channel]).item()

            # Reverse normalization for real-unit losses
            y_norm = norm.reverse(y_processed.cpu().swapaxes(0, 1), "Labels").swapaxes(0, 1)
            y_pred_norm = norm.reverse(y_pred.cpu().swapaxes(0, 1), "Labels").swapaxes(0, 1)

            # Losses in Celsius
            for channel in range(y_pred.shape[1]):
                mae_closs[channel] += L1Loss(reduction="sum")(y_pred_norm[:, channel], y_norm[:, channel]).item()
                rmse_closs[channel] += MSELoss(reduction="sum")(y_pred_norm[:, channel], y_norm[:, channel]).item()

                if vT_case == "temperature":
                    pbt_closs[channel] += (torch.sum(torch.abs(y_pred_norm[:, channel] - y_norm[:, channel]) > pbt_threshold[channel])).item()

        # Calculate metrics
        no_datapoints = len(dataloader.dataset)
        domain_size = y_processed.shape[2] * y_processed.shape[3]
        case_time = time.perf_counter() - case_start_time
        avg_inference_time = total_inference_time / batch_count if batch_count > 0 else 0

        results[case] = {
            "RMSE_normed": [torch.sqrt(mse_loss[i] / (no_datapoints * domain_size)).item() for i in range(output_channels)],
            "MAE_celsius": [mae_closs[i].item() / (no_datapoints * domain_size) for i in range(output_channels)],
            "RMSE_celsius": [torch.sqrt(rmse_closs[i] / (no_datapoints * domain_size)).item() for i in range(output_channels)],
            "timing": {
                "total_time": case_time,
                "total_inference_time": total_inference_time,
                "avg_inference_time_per_batch": avg_inference_time,
                "num_batches": batch_count,
                "num_samples": no_datapoints
            }
        }
        
        if vT_case == "temperature":
            results[case]["PBT_celsius"] = [pbt_closs[i].item() / (no_datapoints * domain_size) for i in range(output_channels)]

        # Log results for this case
        logger.info(f"\n{case.upper()} RESULTS:")
        logger.info(f"  Samples: {no_datapoints}")
        logger.info(f"  RMSE (normalized): {results[case]['RMSE_normed']}")
        logger.info(f"  MAE (°C): {results[case]['MAE_celsius']}")
        logger.info(f"  RMSE (°C): {results[case]['RMSE_celsius']}")
        if vT_case == "temperature":
            logger.info(f"  PBT (°C): {results[case]['PBT_celsius']}")
        logger.info(f"  Total time: {case_time:.2f}s")
        logger.info(f"  Total inference time: {total_inference_time:.2f}s")
        logger.info(f"  Avg inference per batch: {avg_inference_time:.4f}s")
        logger.info(f"  Inference per sample: {total_inference_time/no_datapoints:.4f}s")

    total_time = time.perf_counter() - total_start_time
    logger.info(f"\n" + "=" * 60)
    logger.info(f"MEASUREMENT COMPLETED - Total time: {total_time:.2f}s")
    logger.info("=" * 60)

    return results


def measure_len_width_1K_isoline(data: Dict[str, any]):
    """Measure length and width of 1K isoline."""
    # Import here to avoid circular imports
    from postprocessing.visualization import DataToVisualize
    
    lengths = {}
    widths = {}
    T_gwf = 10.6

    _, axes = plt.subplots(4, 1, sharex=True)
    for index, key in enumerate(["t_true", "t_out"]):
        plt.sca(axes[index])
        datapoint = data[key]
        datapoint.data = torch.flip(datapoint.data, dims=[1])
        left_bound, right_bound = 1280, 0
        upper_bound, lower_bound = 0, 80
        if datapoint.data.max() > T_gwf + 1:
            levels = [T_gwf + 1] 
            CS = plt.contour(datapoint.data.T, levels=levels, cmap='Pastel1', extent=(0,1280,80,0))

            # Calculate maximum width and length of 1K-isoline
            for level in CS.allsegs:
                for seg in level:
                    right_bound = max(right_bound, seg[:,0].max())
                    left_bound = min(left_bound, seg[:,0].min())
                    upper_bound = max(upper_bound, seg[:,1].max())
                    lower_bound = min(lower_bound, seg[:,1].min())
        lengths[key] = max(right_bound - left_bound, 0)
        widths[key] = max(upper_bound - lower_bound, 0)
        print(f"lengths_{key[2:]}.append({lengths[key]})")
        print(f"widths_{key[2:]}.append({widths[key]})")
        print(f"max_temps_{key[2:]}.append({datapoint.data.max()})")
        plt.sca(axes[index+2])
    plt.close("all")
def save_all_measurements(settings: SettingsTraining, len_dataset: int, times: dict, solver: Solver = None, 
                         results: Dict = None, log_file_path: Path = None):
    """Save all measurements to files with enhanced logging."""
    destination = settings.destination
    
    # Setup logger
    if log_file_path is None:
        log_file_path = Path(destination) / "measurements_save.log"
    
    logger = setup_measurement_logger(log_file_path)
    logger.info("Saving measurement results and run information")
    
    # Create info file with run information
    info_dict = {
        "len_dataset": len_dataset,
        "inputs": settings.inputs,
        "case": settings.case,
        "dataset_raw": settings.dataset_raw,
        "problem": settings.problem,
        "epochs": settings.epochs,
        "device": settings.device,
        "notes": settings.notes,
        "times": times,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if solver:
        info_dict.update({
            "best_epoch": solver.best_model_params["epoch"],
            "val_loss": solver.best_model_params["loss"],
            "final_lr": solver.lr_scheduler.get_last_lr()[0] if solver.lr_scheduler else "N/A"
        })
        logger.info(f"Model info - Best epoch: {solver.best_model_params['epoch']}, Val loss: {solver.best_model_params['loss']:.6f}")
    
    # Add measurement results if provided
    if results:
        info_dict["measurement_results"] = results
        logger.info("Measurement results included in saved info")
        
        # Log summary of results
        for case, metrics in results.items():
            logger.info(f"{case} summary:")
            if "timing" in metrics:
                timing = metrics["timing"]
                logger.info(f"  - Total inference time: {timing['total_inference_time']:.2f}s")
                logger.info(f"  - Avg per batch: {timing['avg_inference_time_per_batch']:.4f}s")
                logger.info(f"  - Samples: {timing['num_samples']}")
    
    # Save to YAML file
    info_file_path = destination / "run_info.yaml"
    with open(info_file_path, "w") as f:
        yaml.dump(info_dict, f, default_flow_style=False)
    
    logger.info(f"All measurements saved to: {info_file_path}")
    logger.info(f"Log file saved to: {log_file_path}")
    
    return info_file_path


def _format_results_for_display(results: Dict, vT_case: str) -> Dict:
    """Format measurement results for display."""
    formatted = {}
    for case, metrics in results.items():
        case_results = {
            "RMSE_normed": [f"{val:.2e}" for val in metrics["RMSE_normed"]],
            "MAE_celsius": [f"{val:.2e}" for val in metrics["MAE_celsius"]],
            "RMSE_celsius": [f"{val:.2e}" for val in metrics["RMSE_celsius"]],
        }
        
        if vT_case == "temperature" and "PBT_celsius" in metrics:
            case_results["PBT_celsius"] = [f"{val*100:.2f}%" for val in metrics["PBT_celsius"]]
        
        formatted[case] = case_results
    
    return formatted