import torch
import os
import warnings
import logging

try:
    import pynvml
    HAS_GPU_MONITORING = True
except ImportError:
    HAS_GPU_MONITORING = False

def setup_env():
    """Configura o ambiente para suprimir avisos desnecessários."""
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")
    # ... (restante das configurações de log) ...
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def get_gpu_metrics():
    """Retorna tupla: (VRAM_Usada_MB, Utilizacao_Percentual)"""
    if not HAS_GPU_MONITORING or not torch.cuda.is_available():
        return 0, 0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        # Retorna DOIS valores
        return mem_info.used / 1024**2, utilization.gpu
    except Exception:
        return 0, 0