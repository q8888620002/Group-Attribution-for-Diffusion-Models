"""Utilities"""

import glob
import os
import sys

import pynvml
import torch


def compute_grad_norm(accelerator, model):
    """Compute gradient norm. To be run under the accelerator main process."""
    model = accelerator.unwrap_model(model)
    grads = [
        param.grad.detach().cpu().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    return torch.cat(grads).norm()


def compute_param_norm(accelerator, model):
    """Compute the parameter norm. To be run under the accelerator main process."""
    model = accelerator.unwrap_model(model)
    params = [
        param.data.detach().cpu().flatten()
        for param in model.parameters()
        if param.data is not None
    ]
    return torch.cat(params).norm()


def print_args(args):
    """Print script name and args."""
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")


def get_memory_free_MiB(gpu_index):
    """Method for monitoring GPU usage when debugging."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2


def get_max_step_file(folder_path):
    """Get maximum number of training steps for results in a folder."""
    path_pattern = os.path.join(folder_path, "steps_*.pt")
    files = glob.glob(path_pattern)
    if not files:
        return None
    max_step_file = max(
        files, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
    )
    return max_step_file


def get_max_steps(folder_path):
    """Get maximum number of training steps for results in a folder."""

    path_pattern = os.path.join(folder_path, "ckpt_steps_*.pt")
    files = glob.glob(path_pattern)

    if not files:
        return None

    max_steps = max(
        files, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
    )
    return int(os.path.basename(max_steps).split("_")[-1].split(".")[0])


def get_max_epochs(folder_path):
    """Get maximum number of training steps for results in a folder."""

    path_pattern = os.path.join(folder_path, "ckpt_epochs_*.pt")
    files = glob.glob(path_pattern)

    if not files:
        return None

    max_steps = max(
        files, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
    )
    return int(os.path.basename(max_steps).split("_")[-1].split(".")[0])
