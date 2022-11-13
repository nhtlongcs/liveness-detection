from typing import Any

import torch


def move_to(obj: Any, device: torch.device):
    """Credit: https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283
    Arguments:
        obj {dict, list} -- Object to be moved to device
        device {torch.device} -- Device that object will be moved to
    Raises:
        TypeError: object is of type that is not implemented to process
    Returns:
        type(obj) -- same object but moved to specified device
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        res = {k: move_to(v, device) for k, v in obj.items()}
        return res
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    else:
        try:
            return obj.to(device)
        except:
            raise TypeError(f"Invalid type for move_to {type(obj)}")


def detach(obj: Any):
    """Credit: https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283
    Arguments:
        obj {dict, list} -- Object to be moved to cpu
    Raises:
        TypeError: Invalid type for detach
    Returns:
        type(obj) -- same object but moved to cpu
    """
    if torch.is_tensor(obj):
        return obj.detach()
    elif isinstance(obj, dict):
        res = {k: detach(v) for k, v in obj.items()}
        return res
    elif isinstance(obj, list):
        return [detach(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach(list(obj)))
    else:
        raise TypeError("Invalid type for detach")
