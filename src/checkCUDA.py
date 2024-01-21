import torch

if torch.cuda.is_available():
    print("CUDA está activo en tu entorno virtual de Conda.")
else:
    print("CUDA no está activo en tu entorno virtual de Conda.")

