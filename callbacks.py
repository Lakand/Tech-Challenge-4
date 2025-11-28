import pytorch_lightning as pl
import psutil
import os
import torch

try:
    import pynvml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

class PerformanceMonitorCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Executado automaticamente ao final de cada época de treino.
        """
        # Monitoramento CPU/RAM
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Monitoramento GPU
        gpu_vram = 0
        gpu_util = 0
        
        if HAS_GPU and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_vram = mem.used / 1024**2
                gpu_util = util.gpu
            except:
                pass

        # Logar no MLflow (através do objeto pl_module)
        pl_module.log("sys_ram_mb", ram_mb, prog_bar=True, logger=True)
        pl_module.log("sys_cpu_percent", cpu_percent, prog_bar=False, logger=True)
        pl_module.log("sys_gpu_vram_mb", gpu_vram, prog_bar=True, logger=True)
        pl_module.log("sys_gpu_util_percent", gpu_util, prog_bar=False, logger=True)