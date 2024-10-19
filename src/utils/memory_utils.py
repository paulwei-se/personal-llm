import torch
import gc
import psutil
import logging

logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared")

def get_memory_usage():
    """Get current memory usage statistics"""
    memory_stats = {
        "ram_usage_percent": psutil.virtual_memory().percent,
        "ram_available_gb": psutil.virtual_memory().available / (1024**3)
    }
    
    if torch.cuda.is_available():
        memory_stats.update({
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "gpu_memory_cached_gb": torch.cuda.memory_reserved() / (1024**3)
        })
    
    return memory_stats