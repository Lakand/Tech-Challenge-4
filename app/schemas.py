"""
Esquemas Pydantic.

Define os modelos de dados para validação de entrada e saída da API.
"""
from pydantic import BaseModel

class TrainRequest(BaseModel):
    """
    Parâmetros para requisição de treinamento.
    """
    model_name: str = "default_model"
    symbol: str = "DIS"
    start_date: str = "2018-01-01"
    end_date: str = "2025-11-30"
    epochs: int = 5
    batch_size: int = 32
    prediction_steps: int = 1

class PredictRequest(BaseModel):
    """
    Parâmetros para requisição de inferência.
    """
    model_name: str = "default_model"
    symbol: str = "DIS"
    lookback_days: int = 60