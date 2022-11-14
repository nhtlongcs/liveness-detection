import yaml
import torch


def load_yaml(path):
    with open(path, "rt") as f:
        return yaml.safe_load(f)


def load_state_dict(instance, state_dict, key=None):
    """
    Load trained model checkpoint
    :param model: (nn.Module)
    :param path: (string) checkpoint path
    """

    if isinstance(instance, torch.nn.Module) or isinstance(
            instance, torch.optim.Optimizer):
        try:
            if key is not None:
                instance.load_state_dict(state_dict[key])
            else:
                instance.load_state_dict(state_dict)
            print("Loaded Successfully!")
        except RuntimeError as e:
            print(f"Loaded Successfully. Ignoring {e}")
        return instance
    elif isinstance(instance, dict):
        if key in state_dict.keys():
            return state_dict[key]
        else:
            print(f"Cannot load key={key} from state_dict")
    else:
        raise TypeError(f"{type(instance)} is not supported")
