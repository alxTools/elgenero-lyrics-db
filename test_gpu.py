import bitsandbytes as bnb
import torch

print("bitsandbytes version:", bnb.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
