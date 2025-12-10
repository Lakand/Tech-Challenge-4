import os
import warnings
import logging

def setup_logs_and_warnings():
    """Configura o ambiente para suprimir avisos desnecessários."""
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*pkg_resources.*")
    warnings.filterwarnings("ignore", message=".*local version label.*")
    
    # Silencia bibliotecas barulhentas
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)