"""
Callbacks personalizados para o PyTorch Lightning.

Este módulo define ações que devem ser executadas em momentos específicos do ciclo de vida
do treinamento (ex: ao final de cada época), como o monitoramento de hardware.
"""
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
    """
    Callback para monitorar e registrar o uso de recursos do sistema (CPU, RAM, GPU).

    Este callback é invocado automaticamente pelo PyTorch Lightning ao final de cada
    época de treinamento (`on_train_epoch_end`), coletando métricas em tempo real
    e enviando-as para o MLflow Logger.
    """
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Executado automaticamente ao final de cada época de treino.

        Coleta métricas de hardware e as registra no logger do experimento.

        Args:
            trainer (pl.Trainer): O objeto Trainer que orquestra o treino.
            pl_module (pl.LightningModule): O modelo sendo treinado.
        """
        # Monitoramento CPU/RAM
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Monitoramento GPU (se disponível)
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
            except Exception:
                # Falha silenciosa para não interromper o treino se o driver falhar
                pass

        # Logar no MLflow (através do objeto pl_module)
        pl_module.log("sys_ram_mb", ram_mb, prog_bar=True, logger=True)
        pl_module.log("sys_cpu_percent", cpu_percent, prog_bar=False, logger=True)
        pl_module.log("sys_gpu_vram_mb", gpu_vram, prog_bar=True, logger=True)
        pl_module.log("sys_gpu_util_percent", gpu_util, prog_bar=False, logger=True)