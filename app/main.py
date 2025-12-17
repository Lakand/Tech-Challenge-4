"""
API Principal (FastAPI).

Define os endpoints de treinamento e inferência, gerencia a inicialização
da aplicação e a integração com o MLflow Tracking Server.
"""
import uvicorn
import time
import os
import psutil
import torch
import subprocess
import atexit
import mlflow
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException
from app.config import setup_logs_and_warnings
from app.schemas import TrainRequest, PredictRequest
from app.utils import get_gpu_metrics
from app.services import model_service
from mlflow.tracking import MlflowClient
from pandas.tseries.offsets import BusinessDay

setup_logs_and_warnings()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

app = FastAPI(title="Modular Stock API", description="API para previsão de ações com LSTM e monitoramento MLflow.")

@app.post("/train", summary="Treinar novo modelo")
def train_endpoint(request: TrainRequest):
    """
    Endpoint para iniciar o treinamento de um modelo LSTM.

    Este endpoint:
    1. Baixa os dados históricos conforme solicitado.
    2. Treina o modelo.
    3. Persiste os pesos e metadados (symbol, lookback, features) no disco.
    4. Registra métricas de tempo total e uso de hardware no MLflow.
    """
    start_time = time.time()
    try:
        run_id = model_service.train(request)
        
        duration = time.time() - start_time
        gpu_vram, _ = get_gpu_metrics()
        
        client = MlflowClient()
        client.log_metric(run_id, "train_duration_sec", duration)
        client.log_metric(run_id, "final_gpu_vram_mb", gpu_vram)
        
        return {"message": "Treino concluído", "model": request.model_name, "run_id": run_id}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict", summary="Realizar previsão")
def predict_endpoint(request: PredictRequest):
    """
    Endpoint para inferência de preços futuros.

    Diferente de versões anteriores, este endpoint é inteligente:
    - Recupera automaticamente o Símbolo, Lookback e Features do modelo treinado.
    - Garante que a entrada da predição seja idêntica à usada no treino.
    
    Registra latência, uso de RAM/CPU/GPU e metadados no MLflow.
    """
    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    # 1. Carrega o modelo e seus metadados (symbol, lookback, features)
    if not model_service.load_model(request.model_name):
        raise HTTPException(404, f"Modelo '{request.model_name}' não encontrado.")
    
    model = model_service.current_model
    scaler = model_service.scaler
    current_steps = model_service.prediction_steps
    lookback = model_service.lookback_days
    symbol = model_service.symbol
    features_list = model_service.features 

    if not symbol:
        raise HTTPException(400, "Este modelo é antigo e não possui símbolo salvo. Por favor, treine-o novamente.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    try:
        # 2. Obtém dados recentes baseados no lookback do modelo (não do request)
        import datetime
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=lookback * 4)).strftime("%Y-%m-%d")
        
        df = yf.download(symbol, start=start, end=end)
        
        if len(df) < lookback:
            raise ValueError(f"Dados insuficientes. O modelo requer {lookback} dias, mas só encontramos {len(df)}.")
            
        last_real_date_obj = df.index[-1]
        last_real_date_str = last_real_date_obj.strftime("%Y-%m-%d")
        target_date_obj = last_real_date_obj + BusinessDay(current_steps)
        target_date_str = target_date_obj.strftime("%Y-%m-%d")

        # 3. Garante que as colunas (features) sejam as mesmas do treino
        try:
            input_data = df[features_list].dropna().values[-lookback:]
        except KeyError as e:
            raise ValueError(f"O modelo espera as colunas {features_list}, mas os dados baixados não contêm todas elas. Erro: {e}")

        scaled_input = scaler.transform(input_data)
        
        seq_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(seq_tensor)
            
        dummy = np.zeros((1, model_service.num_features))
        dummy[0, model_service.target_idx] = prediction.cpu().item()
        final_price = scaler.inverse_transform(dummy)[0, model_service.target_idx]
        
        end_time = time.time()
        latency = end_time - start_time
        end_ram = process.memory_info().rss / (1024 * 1024)
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_vram, gpu_util = get_gpu_metrics()
        
        mlflow.set_experiment("TechChallenge_Production_Inference")
        with mlflow.start_run(run_name=f"predict_{request.model_name}"):
            mlflow.log_metric("predicted_price", float(final_price))
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("cpu_usage_percent", cpu_usage)
            mlflow.log_metric("ram_usage_mb", end_ram)
            mlflow.log_metric("gpu_vram_mb", gpu_vram)
            mlflow.log_metric("gpu_util_percent", gpu_util)
            
            mlflow.set_tag("model_used", request.model_name)
            mlflow.set_tag("symbol", symbol)
            mlflow.set_tag("target_date", target_date_str)
            mlflow.set_tag("prediction_horizon", current_steps)

        return {
            "model_used": request.model_name,
            "symbol": symbol,
            "predicted_close_price": float(final_price),
            "prediction_info": {
                "steps_ahead": current_steps,
                "target_date": target_date_str
            },
            "performance": {
                "latency_sec": round(latency, 4),
                "ram_usage_mb": round(end_ram, 2),
                "cpu_usage_percent": cpu_usage,
                "gpu_vram_mb": round(gpu_vram, 2),
                "gpu_util_percent": gpu_util
            }
        }

    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    mlflow_process = subprocess.Popen(["mlflow", "ui", "--port", "5000"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(lambda: subprocess.run(f"taskkill /F /PID {mlflow_process.pid} /T", shell=True, stderr=subprocess.DEVNULL))
    
    print("API Modular Iniciada!")
    uvicorn.run(app, host="0.0.0.0", port=8000)