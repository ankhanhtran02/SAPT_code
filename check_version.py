import datasets
#check version of datasets library
print(datasets.__version__)




import torch
from peft import PeftModel, PeftConfig
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version used by PyTorch:", torch.version.cuda)
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
