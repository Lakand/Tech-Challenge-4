"""
Esquemas Pydantic.

Define os modelos de dados para validação de entrada e saída da API.
"""
from pydantic import BaseModel

class TrainRequest(BaseModel):
    """
    Parâmetros para requisição de treinamento.

    Define a configuração do treino e a arquitetura temporal do modelo (lookback).
    """
    model_name: str = "disney_v1"
    symbol: str = "DIS"
    start_date: str = "2018-01-01"
    end_date: str = "2025-11-30"
    epochs: int = 5
    batch_size: int = 32
    prediction_steps: int = 1
    lookback_days: int = 60

class PredictRequest(BaseModel):
    """
    Parâmetros para requisição de inferência.

    Solicita apenas o nome do modelo, pois o símbolo e o lookback são
    inferidos automaticamente a partir dos metadados do modelo treinado.
    """
    model_name: str = "disney_v1"