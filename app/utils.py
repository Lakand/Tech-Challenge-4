"""
Utilitários de Sistema e Hardware.

Contém funções auxiliares para interagir com o sistema operativo e drivers de hardware,
como a detecção e leitura de métricas da placa gráfica (NVIDIA GPU).
"""
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
    """
    Configura o ambiente de execução Python.

    Define variáveis de ambiente e filtros de avisos (warnings) para garantir
    que a saída do console permaneça limpa e legível, suprimindo mensagens
    de depreciação irrelevantes das bibliotecas.
    """
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")
    # Silencia logs verbosos de bibliotecas de ML
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def get_gpu_metrics():
    """
    Obtém métricas instantâneas da GPU NVIDIA primária.

    Utiliza a biblioteca pynvml (bindings Python para NVML) para ler o estado da GPU.
    Se não houver GPU ou a biblioteca não estiver instalada, retorna zeros.

    Returns:
        tuple: (VRAM_Usada_MB, Utilizacao_Percentual)
               Exemplo: (2048.5, 45)
    """
    if not HAS_GPU_MONITORING or not torch.cuda.is_available():
        return 0, 0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # Retorna memória usada em MB e porcentagem de uso do núcleo
        return mem_info.used / 1024**2, utilization.gpu
    except Exception:
        return 0, 0