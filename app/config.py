"""
Configurações Globais da Aplicação.

Centraliza as configurações de logging e supressão de avisos para serem
reutilizadas em diferentes pontos de entrada da aplicação.
"""
import os
import warnings
import logging

def setup_logs_and_warnings():
    """
    Aplica as políticas de log e avisos do projeto.

    Esta função deve ser chamada no início da execução (ex: no main.py) antes
    de qualquer outra importação pesada para garantir que os avisos de inicialização
    sejam capturados ou silenciados conforme desejado.
    """
    os.environ["PYTHONWARNINGS"] = "ignore"
    
    # Filtra avisos de depreciação comuns em bibliotecas de Data Science (Pandas, PyTorch)
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*pkg_resources.*")
    warnings.filterwarnings("ignore", message=".*local version label.*")
    
    # Define o nível de log para ERROR, evitando poluição visual com INFO/DEBUG
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)