"""Optional CPU storage for tensors retained by autograd, including checkpoints."""
from contextlib import nullcontext

import torch


def saved_tensor_offloading(enabled: bool):
    # pin_memory enables asynchronous transfers back to the original CUDA device.
    # torch's context restores dtype/device and also supports checkpoint replay.
    return torch.autograd.graph.save_on_cpu(pin_memory=True) if enabled else nullcontext()
