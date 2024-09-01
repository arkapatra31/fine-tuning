import torch

# Set device to CPU
#device = torch.device("cpu")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

__all__ = [
    device
]