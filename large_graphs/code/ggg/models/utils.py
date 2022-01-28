import torch
from ggg.models.components.utilities_classes import Swish


def get_act(swish):
    if swish is True or swish == "swish":
        act = Swish
    elif swish == "leaky":
        act = lambda: torch.nn.LeakyReLU(0.1)
    elif swish == "celu":
        act = torch.nn.CELU
    else:
        act = torch.nn.ReLU
    return act