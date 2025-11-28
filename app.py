# --- BLOCO 1: LIMPEZA E CONFIGURAÇÃO ---
import os
import warnings
import logging

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*local version label.*")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# --- BLOCO 2: IMPORTS ---
import torch
import joblib
import time
import psutil
import subprocess
import atexit
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from mlflow.models.signature import infer_signature
from pandas.tseries.offsets import BusinessDay
from callbacks import PerformanceMonitorCallback
from mlflow.tracking import MlflowClient

try:
    import pynvml
    HAS_GPU_MONITORING = True
except ImportError:
    HAS_GPU_MONITORING = False

from model import StockLSTM
from dataset import get_dataloaders

# Cria pasta para salvar os modelos se não existir
os.makedirs("models", exist_ok=True)

app = FastAPI(title="Multi-Model Stock API")

# --- BLOCO 3: GESTÃO DE ESTADO GLOBAL ---
# Agora guardamos o NOME do modelo carregado atualmente
current_model_name = None
current_model = None
current_scaler = None
current_num_features = 5
current_target_idx = 3
current_prediction_steps = 1

def smart_load_model(model_name: str):
    """
    Carrega o modelo do disco APENAS se ele for diferente do que já está na memória.
    """
    global current_model, current_scaler, current_num_features, current_target_idx, current_prediction_steps, current_model_name
    
    # Se já estamos com esse modelo carregado, não faz nada (Cache)
    if current_model_name == model_name and current_model is not None:
        return True

    # Define os caminhos baseados no nome
    model_path = f"models/{model_name}.pth"
    artifacts_path = f"models/{model_name}.pkl"

    if os.path.exists(model_path) and os.path.exists(artifacts_path):
        try:
            print(f"🔄 Trocando modelo para: {model_name}...")
            artifacts = joblib.load(artifacts_path)
            
            # Atualiza globais
            current_scaler = artifacts['scaler']
            current_num_features = artifacts['num_features']
            current_target_idx = artifacts['target_idx']
            current_prediction_steps = artifacts.get('prediction_steps', 1)
            
            # Recria arquitetura e carrega pesos
            model = StockLSTM(input_dim=current_num_features)
            model.load_state_dict(torch.load(model_path))
            
            # Move para GPU se disponível
            if torch.cuda.is_available():
                model.cuda()
            else:
                model.eval()
            
            current_model = model
            current_model_name = model_name
            
            if __name__ == "__main__":
                print(f"✅ Modelo '{model_name}' carregado com sucesso!")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo '{model_name}': {e}")
            return False
    else:
        print(f"❌ Modelo '{model_name}' não encontrado no disco.")
        return False

# --- SCHEMAS (Atualizados com model_name) ---
class TrainRequest(BaseModel):
    model_name: str = "default_model" # <--- O NOME DO ARQUIVO VAI AQUI
    symbol: str = "DIS"
    start_date: str = "2018-01-01"
    end_date: str = "2024-07-20"
    epochs: int = 5
    batch_size: int = 32
    prediction_steps: int = 1

class PredictRequest(BaseModel):
    model_name: str = "default_model" # <--- QUAL MODELO USAR?
    symbol: str # Pode ser opcional se você quiser forçar o simbolo do treino, mas deixaremos flexível
    lookback_days: int = 60

# --- FUNÇÃO AUXILIAR GPU ---
def get_gpu_metrics():
    if not HAS_GPU_MONITORING or not torch.cuda.is_available():
        return 0, 0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return mem_info.used / 1024**2, utilization.gpu
    except Exception:
        return 0, 0

# --- ROTAS ---

@app.post("/train")
def train_model(request: TrainRequest):
    # --- 1. INÍCIO DO MONITORAMENTO (Snapshot) ---
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_ram = process.memory_info().rss / (1024 * 1024)
    
    global current_model, current_scaler, current_num_features, current_target_idx, current_prediction_steps, current_model_name
    
    mlf_logger = MLFlowLogger(experiment_name="TechChallenge_Training", tracking_uri="http://localhost:5000")
    
    # Log de Hiperparâmetros (mantém igual)
    mlf_logger.log_hyperparams({
        "model_name": request.model_name,
        "symbol": request.symbol,
        "steps": request.prediction_steps,
        "epochs": request.epochs,
        "batch_size": request.batch_size
    })
    
    try:
        train_loader, val_loader, scaler, num_features, target_idx = get_dataloaders(
            request.symbol, 
            request.start_date, 
            request.end_date, 
            batch_size=request.batch_size,
            prediction_steps=request.prediction_steps 
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    current_scaler = scaler
    current_num_features = num_features
    current_target_idx = target_idx
    current_prediction_steps = request.prediction_steps
    current_model_name = request.model_name

    model = StockLSTM(input_dim=num_features)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else "auto"
    
    # Inicializa Trainer
    trainer = pl.Trainer(max_epochs=request.epochs,
                         logger=mlf_logger,
                         accelerator=accelerator,
                         devices=devices,
                         callbacks=[PerformanceMonitorCallback()])
    
    # TREINAMENTO ACONTECE AQUI
    trainer.fit(model, train_loader, val_loader)
    
    current_model = model

    # Salvamento (mantém igual)
    save_path_model = f"models/{request.model_name}.pth"
    save_path_artifacts = f"models/{request.model_name}.pkl"
    torch.save(model.state_dict(), save_path_model)
    joblib.dump({
        'scaler': scaler,
        'num_features': num_features,
        'target_idx': target_idx,
        'prediction_steps': request.prediction_steps
    }, save_path_artifacts)
    
    # --- 2. FIM DO MONITORAMENTO ---
    end_time = time.time()
    total_training_time = end_time - start_time
    
    end_ram = process.memory_info().rss / (1024 * 1024)
    cpu_usage = psutil.cpu_percent(interval=None) # Média instantânea no final
    gpu_vram_mb, gpu_util_percent = get_gpu_metrics()
    
    client = MlflowClient()
    run_id = trainer.logger.run_id

    # Logar Métricas de Performance do Treino no MLflow
    # Nota: Usamos o run_id do trainer para adicionar essas métricas na mesma run do treinamento
    with mlflow.start_run(run_id=trainer.logger.run_id):
        mlflow.log_metric("train_total_time_sec", total_training_time)
        mlflow.log_metric("train_final_ram_mb", end_ram)
        mlflow.log_metric("train_final_cpu_percent", cpu_usage)
        mlflow.log_metric("train_final_gpu_vram_mb", gpu_vram_mb)
    
    return {
        "message": f"Treinamento de '{request.model_name}' concluído.",
        "performance": {
            "duration_sec": round(total_training_time, 2),
            "final_ram_mb": round(end_ram, 2),
            "gpu_vram_mb": round(gpu_vram_mb, 2)
        },
        "files_saved": [save_path_model, save_path_artifacts]
    }

@app.post("/predict")
def predict(request: PredictRequest):
    # --- 1. INÍCIO DO MONITORAMENTO (Snapshot inicial) ---
    start_time = time.time()
    process = psutil.Process(os.getpid())
    # Captura uso inicial (opcional, mas bom para comparar delta se quiser)
    start_ram = process.memory_info().rss / (1024 * 1024) 
    
    # 2. Carregamento Inteligente
    if not smart_load_model(request.model_name):
        raise HTTPException(status_code=404, detail=f"Modelo '{request.model_name}' não encontrado.")
    
    # Prepara dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_model.to(device)
    current_model.eval()
    
    try:
        import datetime
        
        # 3. Lógica de Dados e Previsão
        end = datetime.datetime.now().strftime("%Y-%m-%d")
        start = (datetime.datetime.now() - datetime.timedelta(days=request.lookback_days * 4)).strftime("%Y-%m-%d")
        
        df = yf.download(request.symbol, start=start, end=end)
        
        if len(df) < request.lookback_days:
            raise ValueError("Dados insuficientes.")
            
        # Datas
        last_real_date_obj = df.index[-1]
        last_real_date_str = last_real_date_obj.strftime("%Y-%m-%d")
        target_date_obj = last_real_date_obj + BusinessDay(current_prediction_steps)
        target_date_str = target_date_obj.strftime("%Y-%m-%d")

        # Inferência
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        input_data = df[features].dropna().values[-request.lookback_days:]
        scaled_input = current_scaler.transform(input_data)
        
        seq_tensor = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = current_model(seq_tensor)
            
        dummy_array = np.zeros((1, current_num_features))
        dummy_array[0, current_target_idx] = prediction.cpu().item()
        final_price = current_scaler.inverse_transform(dummy_array)[0, current_target_idx]
        
        # --- 4. FIM DO MONITORAMENTO (Coleta de Métricas) ---
        end_time = time.time()
        
        # Latência (Tempo total de execução)
        latency = end_time - start_time
        
        # Memória RAM usada pelo processo Python agora
        end_ram = process.memory_info().rss / (1024 * 1024) # Em MB
        
        # Uso de CPU (Percentual instantâneo)
        cpu_usage = psutil.cpu_percent(interval=None)
        
        # Métricas de GPU (Função auxiliar que criamos)
        gpu_vram_mb, gpu_util_percent = get_gpu_metrics()
        
        # --- 5. LOG COMPLETO NO MLFLOW ---
        mlflow.set_experiment("TechChallenge_Production_Inference")
        
        with mlflow.start_run(run_name=f"predict_{request.model_name}"):
            # A. Métricas de Negócio
            mlflow.log_metric("predicted_price", float(final_price))
            
            # B. Métricas de Performance (O que estava faltando)
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("cpu_usage_percent", cpu_usage)
            mlflow.log_metric("ram_usage_mb", end_ram)
            mlflow.log_metric("gpu_vram_mb", gpu_vram_mb)
            mlflow.log_metric("gpu_util_percent", gpu_util_percent)
            
            # C. Tags de Contexto (Metadata)
            mlflow.set_tag("model_used", request.model_name)
            mlflow.set_tag("symbol", request.symbol)
            mlflow.set_tag("target_date", target_date_str)
            mlflow.set_tag("prediction_horizon", current_prediction_steps)
        
        # Retorno da API (Também mostra no JSON para fácil conferência)
        return {
            "model_used": request.model_name,
            "symbol": request.symbol,
            "predicted_close_price": float(final_price),
            "prediction_info": {
                "steps_ahead": current_prediction_steps,
                "target_date": target_date_str
            },
            "performance": {
                "latency_sec": round(latency, 4),
                "ram_usage_mb": round(end_ram, 2),
                "cpu_usage_percent": cpu_usage,
                "gpu_vram_mb": round(gpu_vram_mb, 2),
                "gpu_util_percent": gpu_util_percent
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Inicializa MLflow em segundo plano
    mlflow_process = subprocess.Popen(["mlflow", "ui", "--port", "5000"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def cleanup():
        subprocess.run(f"taskkill /F /PID {mlflow_process.pid} /T", shell=True, stderr=subprocess.DEVNULL)
    atexit.register(cleanup)
    
    print("✅ MLflow rodando em http://localhost:5000")
    print("✅ API iniciando em http://localhost:8000")

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        pass # Permite sair sem erro feio na tela