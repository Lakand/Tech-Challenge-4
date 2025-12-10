from pydantic import BaseModel

class TrainRequest(BaseModel):
    model_name: str = "default_model"
    symbol: str = "DIS"
    start_date: str = "2018-01-01"
    end_date: str = "2024-07-20"
    epochs: int = 5
    batch_size: int = 32
    prediction_steps: int = 1

class PredictRequest(BaseModel):
    model_name: str = "default_model"
    symbol: str
    lookback_days: int = 60