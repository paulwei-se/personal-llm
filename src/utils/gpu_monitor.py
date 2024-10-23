import torch
import threading
import time
from typing import Dict, Optional
import logging
from pynvml import *

logger = logging.getLogger(__name__)

class GPUMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.running = False
        self.stats: Dict[str, float] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
            self.nvml_available = True
        except Exception as e:
            logger.warning(f"NVML initialization failed: {e}")
            self.nvml_available = False
            
    def start(self):
        self.running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("GPU monitoring started")
        
    def stop(self):
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("GPU monitoring stopped")
        
    def _monitor_loop(self):
        while self.running:
            self.stats = self.get_stats()
            time.sleep(self.interval)
            
    def get_stats(self) -> Dict[str, float]:
        stats = {
            'memory_used': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved': torch.cuda.memory_reserved() / 1024**3
        }
        
        if self.nvml_available:
            try:
                info = nvmlDeviceGetMemoryInfo(self.handle)
                utilization = nvmlDeviceGetUtilizationRates(self.handle)
                stats.update({
                    'gpu_utilization': utilization.gpu,
                    'memory_total': info.total / 1024**3,
                    'memory_percentage': (info.used / info.total) * 100
                })
            except Exception as e:
                logger.warning(f"Error getting NVML stats: {e}")
        
        return stats

    def __del__(self):
        if self.nvml_available:
            try:
                nvmlShutdown()
            except:
                pass