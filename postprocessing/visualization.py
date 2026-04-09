import time
from dataclasses import dataclass, field
from math import inf
from typing import Dict

# import matplotlib as mpl
# mpl.use('pgf')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader

from data_stuff.transforms import NormalizeTransform
from networks.unet import UNet
import processing.rotation as rt

# mpl.rcParams.update({'figure.max_open_warning': 0})
# plt.rcParams['figure.figsize'] = [16, 5]

# TODO: look at vispy library for plotting 3D data


def _get_dataset_attr(dataset, attr_name: str):
    """Walk through nested dataset wrappers and return the first matching attribute."""
    current = dataset
    while current is not None:
        if hasattr(current, attr_name):
            return getattr(current, attr_name)
        current = getattr(current, "dataset", None)
    raise AttributeError(
        f"Could not find attribute '{attr_name}' on dataset or wrapped datasets."
    )


@dataclass
class DataToVisualize:
    data: np.ndarray
    name: str
    extent_highs: tuple = (1280, 100)  # x,y in meters
    imshowargs: Dict = field(default_factory=dict)
    contourfargs: Dict = field(default_factory=dict)
    contourargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        extent = (0, int(self.extent_highs[0]), int(self.extent_highs[1]), 0)

        self.imshowargs = {"cmap": "RdBu_r", "extent": extent}

        self.contourfargs = {
            "levels": np.arange(10.4, 16, 0.25),
            "cmap": "RdBu_r",
            "extent": extent,
        }

        T_gwf = 10.6
        T_inj_diff = 5.0
        self.contourargs = {
            "levels": [np.round(T_gwf + 1, 1)],
            "cmap": "Pastel1",
            "extent": extent,
        }

        if self.name == "Liquid Pressure [Pa]":
            self.name = "Pressure in [Pa]"
        elif self.name == "Material ID":
            self.name = "Position of the heatpump in [-]"
        elif self.name == "Permeability X [m^2]":
            self.name = "Permeability in [m$^2$]"
        elif self.name == "SDF":
            self.name = "SDF-transformed position in [-]"


def visualizations(
    model: UNet,
    dataloader: DataLoader,
    device: str,
    amount_datapoints_to_visu: int = inf,
    plot_path: str = "default",
    pic_format: str = "png",
    rotate_inference: bool = False,
    crop: bool = False,
):
    print("Visualizing...", end="\r")

    if amount_datapoints_to_visu > len(dataloader.dataset):
        amount_datapoints_to_visu = len(dataloader.dataset)

    norm = _get_dataset_attr(dataloader.dataset, "norm")
    info = _get_dataset_attr(dataloader.dataset, "info")
    model.eval()
    settings_pic = {
        "format": pic_format,
        "dpi": 600,
    }

    current_id = 0
    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            name_pic = f"{plot_path}_{current_id}"

            x = torch.unsqueeze(inputs[datapoint_id].to(device), 0)
            y = labels[datapoint_id]
            y = rt.safe_center_crop(y, rt.get_safe_size(y)) if crop else y

            # rotate data point if Oriented Boxes approach is used
            if rotate_inference:
                y_out, angle = rt.rotate_and_infer(
                    x.squeeze(0), [-1, 0], model, info, device, crop, return_angle=True
                )
                y_out = y_out.to(device)

                # We also rotate and crop the ground truth to match the aligned prediction
                y_aligned = rt.rotate(y, angle)
                if crop:
                    y_aligned = rt.safe_center_crop(
                        y_aligned, rt.get_safe_size(y_aligned)
                    )
                y = y_aligned

                # Rotate and crop the input so visualization aligns with prediction
                x_aligned = rt.rotate(x, angle)
                x = (
                    rt.safe_center_crop(
                        x_aligned.cpu(), rt.get_safe_size(x_aligned.cpu())
                    ).to(device)
                    if crop
                    else x_aligned.to(device)
                )
            else:
                x = (
                    rt.safe_center_crop(x.cpu(), rt.get_safe_size(x.cpu())).to(device)
                    if crop
                    else x
                )
                y_out = model(x).to(device)

            x, y, y_out = reverse_norm_one_dp(x, y, y_out, norm)
            dict_to_plot = prepare_data_to_plot(x, y, y_out, info)

            np.save(
                f"{plot_path}_label_{current_id}.npy",
                dict_to_plot["t_true"].data.T.numpy(),
            )
            np.save(
                f"{plot_path}_prediction_{current_id}.npy",
                dict_to_plot["t_out"].data.T.numpy(),
            )

            plot_datafields(dict_to_plot, name_pic, settings_pic)
            # plot_isolines(dict_to_plot, name_pic, settings_pic)
            # measure_len_width_1K_isoline(dict_to_plot)

            if current_id >= amount_datapoints_to_visu - 1:
                return None
            current_id += 1


def reverse_norm_one_dp(
    x: torch.Tensor, y: torch.Tensor, y_out: torch.Tensor, norm: NormalizeTransform
):
    # reverse transform for plotting real values
    x = norm.reverse(x.detach().cpu().squeeze(0), "Inputs")
    y = norm.reverse(y.detach().cpu(), "Labels")[0]
    y_out = norm.reverse(y_out.detach().cpu()[0], "Labels")[0]
    return x, y, y_out


def prepare_data_to_plot(
    x: torch.Tensor, y: torch.Tensor, y_out: torch.Tensor, info: dict
):
    # prepare data of temperature true, temperature out, error, physical variables (inputs)
    temp_max = max(y.max(), y_out.max())
    temp_min = min(y.min(), y_out.min())
    extent_highs = np.array(info["CellsSize"][:2]) * x.shape[-2:]

    dict_to_plot = {
        "t_true": DataToVisualize(
            y,
            "Label: Temperature in [°C]",
            extent_highs,
            {"vmax": temp_max, "vmin": temp_min},
        ),
        "t_out": DataToVisualize(
            y_out,
            "Prediction: Temperature in [°C]",
            extent_highs,
            {"vmax": temp_max, "vmin": temp_min},
        ),
        "error": DataToVisualize(
            torch.abs(y - y_out), "Absolute error in [°C]", extent_highs
        ),
    }
    inputs = info["Inputs"].keys()
    for input in inputs:
        index = info["Inputs"][input]["index"]
        dict_to_plot[input] = DataToVisualize(x[index], input, extent_highs)

    return dict_to_plot


def plot_datafields(
    data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict
):
    # plot datafields (temperature true, temperature out, error, physical variables (inputs))
    fontsize = 8
    num_subplots = len(data)
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)

    for index, (name, datapoint) in enumerate(data.items()):
        plt.sca(axes[index])
        plt.title(datapoint.name, fontsize=fontsize, pad=10)
        # if name in ["t_true", "t_out"]:
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")

        #         CS = plt.contour(torch.flip(datapoint.data, dims=[1]).T, **datapoint.contourargs)
        #     plt.clabel(CS, inline=1, fontsize=10)
        plt.imshow(datapoint.data.T, **datapoint.imshowargs)
        plt.gca().invert_yaxis()

        plt.ylabel("x [m]", fontsize=fontsize)
        plt.tick_params(axis="both", labelsize=fontsize)

        _aligned_colorbar(fontsize=fontsize)
        plt.tick_params(axis="both", labelsize=fontsize)

    plt.sca(axes[-1])
    plt.xlabel("y [m]", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{name_pic}.{settings_pic['format']}", **settings_pic)


def plot_isolines(data: Dict[str, DataToVisualize], name_pic: str, settings_pic: dict):
    # plot isolines of temperature fields
    num_subplots = 3 if "Original Temperature [C]" in data.keys() else 2
    fig, axes = plt.subplots(num_subplots, 1, sharex=True)
    fig.set_figheight(num_subplots)

    for index, name in enumerate(["t_true", "t_out", "Original Temperature [C]"]):
        try:
            plt.sca(axes[index])
            data[name].data = torch.flip(data[name].data, dims=[1])
            plt.title("Isolines of " + data[name].name)
            plt.contourf(data[name].data.T, **data[name].contourfargs)
            plt.ylabel("x [m]")
            _aligned_colorbar(ticks=[11.6, 15.6])
        except:
            pass

    plt.sca(axes[-1])
    plt.xlabel("y [m]")
    plt.tight_layout()
    plt.savefig(f"{name_pic}_isolines.{settings_pic['format']}", **settings_pic)


def infer_all_and_summed_pic(
    model: UNet,
    dataloader: DataLoader,
    device: str,
    rotate_inference: bool = False,
    angle: int = 0,
    crop: bool = False,
):
    """
    sum inference time (including reverse-norming) and pixelwise error over all datapoints
    the angle parameter is only used for testing of equivariance
    """

    norm = _get_dataset_attr(dataloader.dataset, "norm")
    info = _get_dataset_attr(dataloader.dataset, "info")
    model.eval()

    current_id = 0
    avg_inference_time = 0
    summed_error_pic = None

    for inputs, labels in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            # get data
            # start_time = time.perf_counter()
            x = rt.rotate(inputs[datapoint_id], angle).to(device)
            x = torch.unsqueeze(x, 0)

            # rotate data point if Oriented Boxes approach is used
            if rotate_inference:
                start_time = time.perf_counter()
                y_out, infer_angle = rt.rotate_and_infer(
                    x.squeeze(0), [-1, 0], model, info, device, crop, return_angle=True
                )
                y_out = y_out.to(device)

                # Rotate y forward instead of evaluating in original space
                y_aligned = rt.rotate(
                    labels[datapoint_id], angle
                )  # apply prior rotation (equivariance test angle)
                y_aligned = rt.rotate(
                    y_aligned, infer_angle
                )  # apply alignment rotation
                if crop:
                    y_aligned = rt.safe_center_crop(
                        y_aligned, rt.get_safe_size(y_aligned)
                    )
                y = y_aligned

                x_aligned = rt.rotate(x, infer_angle)
                x = (
                    rt.safe_center_crop(
                        x_aligned.cpu(), rt.get_safe_size(x_aligned.cpu())
                    ).to(device)
                    if crop
                    else x_aligned.to(device)
                )
            else:
                x = (
                    rt.safe_center_crop(x.cpu(), rt.get_safe_size(x.cpu())).to(device)
                    if crop
                    else x
                )
                start_time = time.perf_counter()
                y_out = model(x).to(device)

                y = rt.rotate(labels[datapoint_id], angle)
                y = rt.safe_center_crop(y, rt.get_safe_size(y)) if crop else y

            avg_inference_time += time.perf_counter() - start_time

            # reverse transform for plotting real values
            x = norm.reverse(x.cpu().detach().squeeze(), "Inputs")
            y = norm.reverse(y.cpu().detach(), "Labels")[0]
            y_out = norm.reverse(y_out.cpu().detach()[0], "Labels")[0]

            if summed_error_pic is None:
                summed_error_pic = torch.zeros_like(y_out).cpu()

            # avg_inference_time += (time.perf_counter() - start_time)
            summed_error_pic += abs(y - y_out)

            current_id += 1

    avg_inference_time /= current_id
    summed_error_pic /= current_id

    if rotate_inference:
        return avg_inference_time, summed_error_pic
    else:
        return avg_inference_time, rt.rotate(
            summed_error_pic.unsqueeze(0), 360 - angle
        ).squeeze(0)


def infer_all_rotate_and_summed_pic(
    model: UNet,
    dataloader: DataLoader,
    device: str,
    rotate_inference: bool = False,
    angle: int = 0,
    crop: bool = False,
):
    """
    sum inference time (including reverse-norming)
    pixelwise error between all datapoints and all rotated datapoints
    """

    norm = _get_dataset_attr(dataloader.dataset, "norm")
    info = _get_dataset_attr(dataloader.dataset, "info")
    model.eval()

    current_id = 0
    summed_error_pic = None

    for inputs, _ in dataloader:
        len_batch = inputs.shape[0]
        for datapoint_id in range(len_batch):
            # get data
            x = inputs[datapoint_id].to(device)
            x = torch.unsqueeze(x, 0)
            if crop:
                x = rt.safe_center_crop(x.cpu(), rt.get_safe_size(x.cpu())).to(device)

            # get rotated data
            x_rot = rt.rotate(inputs[datapoint_id], angle).to(device)
            x_rot = torch.unsqueeze(x_rot, 0)
            if crop:
                x_rot = rt.safe_center_crop(
                    x_rot.cpu(), rt.get_safe_size(x_rot.cpu())
                ).to(device)

            # get inference for rotated and unrotated data
            if rotate_inference:
                # To compare them without artifacts, we align both to the flow direction!
                y_out, infer_angle = rt.rotate_and_infer(
                    x.squeeze(0), [-1, 0], model, info, device, crop, return_angle=True
                )
                y_out = y_out.to(device)

                y_out_rot, infer_angle_rot = rt.rotate_and_infer(
                    x_rot.squeeze(0),
                    [-1, 0],
                    model,
                    info,
                    device,
                    crop,
                    return_angle=True,
                )
                y_out_rot = y_out_rot.to(device)

                # They are both aligned to [-1, 0]. But their original starting orientations differed by `angle`.
                # Because the physical situation is the same (just rotated), their ALIGNED predictions
                # should ideally be identical! So we can just compare them directly in aligned space.
            else:
                y_out = model(x).to(device)
                y_out_rot = model(x_rot).to(device)
                # rotate prediction for rotated data back
                y_out_rot = rt.rotate(y_out_rot, 360 - angle)

            # reverse transform for plotting real values
            y_out_rot = norm.reverse(y_out_rot.cpu().detach()[0], "Labels")[0]
            y_out = norm.reverse(y_out.cpu().detach()[0], "Labels")[0]

            if summed_error_pic is None:
                summed_error_pic = torch.zeros_like(y_out).cpu()

            # calculate error between inference of rotated and unrotated input
            summed_error_pic += abs(y_out - y_out_rot)

            current_id += 1

    summed_error_pic /= current_id
    return summed_error_pic


def plot_avg_error_rotated_cellwise(
    dataloader, summed_error_pic_dif, settings_pic: dict, angle: int
):
    # plot avg error cellwise between predictions of rotated and unrotated inputs

    info = _get_dataset_attr(dataloader.dataset, "info")
    extent_highs = np.array(info["CellsSize"][:2]) * dataloader.dataset[0][0][0].shape
    extent = (0, int(extent_highs[0]), int(extent_highs[1]), 0)

    plt.figure()
    plt.imshow(summed_error_pic_dif.T, cmap="RdBu_r", extent=extent)
    plt.gca().invert_yaxis()
    plt.ylabel("x [m]")
    plt.xlabel("y [m]")
    plt.title("Difference cellwise averaged error [°C]")
    _aligned_colorbar()

    plt.tight_layout()
    plt.savefig(
        f"{settings_pic['folder']}/avg_error_dif_{angle}.{settings_pic['format']}",
        format=settings_pic["format"],
    )


def plot_avg_error_cellwise(dataloader, summed_error_pic, settings_pic: dict):
    # plot avg error cellwise AND return time measurements for inference

    info = _get_dataset_attr(dataloader.dataset, "info")
    extent_highs = np.array(info["CellsSize"][:2]) * dataloader.dataset[0][0][0].shape
    extent = (0, int(extent_highs[0]), int(extent_highs[1]), 0)

    plt.figure()
    plt.imshow(summed_error_pic.T, cmap="RdBu_r", extent=extent)
    plt.gca().invert_yaxis()
    plt.ylabel("x [m]")
    plt.xlabel("y [m]")
    plt.title("Cellwise averaged error [°C]")
    _aligned_colorbar()

    plt.tight_layout()
    plt.savefig(
        f"{settings_pic['folder']}/avg_error.{settings_pic['format']}",
        format=settings_pic["format"],
    )


def _aligned_colorbar(fontsize: int = -1, *args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes("right", size=0.15, pad=0.05)
    if fontsize > 0:
        offset_text = cax.yaxis.get_offset_text()
        offset_text.set_fontsize(fontsize)
    plt.colorbar(*args, cax=cax, **kwargs)
