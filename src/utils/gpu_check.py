# src/utils/gpu_check.py
import torch

def check_gpu():
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "No GPU"
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    
    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "current_device": torch.cuda.current_device() if cuda_available else None,
        "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB" if cuda_available else "N/A"
    }

# Test directly
if __name__ == "__main__":
    print(check_gpu())