import torch

# 打印 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 检查 MPS (Apple Silicon GPU) 后端是否可用
print(f"Is MPS backend available? {torch.backends.mps.is_available()}")

# 检查 PyTorch 在构建时是否包含了 MPS 支持
print(f"Is MPS backend built? {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Great! MPS device is available. We can use the GPU!")
    try:
        # 尝试在 MPS 设备上创建一个张量
        x = torch.ones(1, device=device)
        print("Successfully created a tensor on the MPS device.")
        print("Your environment seems to be set up correctly for GPU usage!")
    except Exception as e:
        print(f"An error occurred while trying to use the MPS device: {e}")
else:
    print("Unfortunately, MPS device is not available. PyTorch will fall back to CPU.")
    print("This might be due to an incorrect PyTorch installation or version incompatibility.")