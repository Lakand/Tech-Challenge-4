"""
Serviços de Orquestração de Machine Learning.

Este módulo contém a lógica de negócio para gerenciar o ciclo de vida dos modelos,
incluindo treinamento, carregamento de artefatos e gestão de estado global (Singleton).
"""
import os
import torch
import joblib
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from ml.model import StockLSTM
from ml.dataset import get_dataloaders
from ml.callbacks import PerformanceMonitorCallback

class ModelService:
    """
    Gerenciador de Modelos (Singleton).

    Responsável por manter o modelo carregado em memória para inferência
    e coordenar o processo de treinamento assíncrono.
    """
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        self.scaler = None
        self.num_features = 5
        self.target_idx = 3
        self.prediction_steps = 1
        
        # Garante a existência da pasta de persistência
        os.makedirs("models", exist_ok=True)

    def load_model(self, model_name: str) -> bool:
        """
        Carrega um modelo e seus artefatos do disco para a memória.

        Verifica se o modelo solicitado já está em memória (cache) para evitar I/O desnecessário.

        Args:
            model_name (str): Nome do modelo a ser carregado (sem extensão).

        Returns:
            bool: True se carregado com sucesso (ou já existente), False caso contrário.
        """
        if self.current_model_name == model_name and self.current_model is not None:
            return True

        model_path = f"models/{model_name}.pth"
        artifacts_path = f"models/{model_name}.pkl"

        if os.path.exists(model_path) and os.path.exists(artifacts_path):
            try:
                print(f"🔄 Carregando modelo: {model_name}...")
                artifacts = joblib.load(artifacts_path)
                
                self.scaler = artifacts['scaler']
                self.num_features = artifacts['num_features']
                self.target_idx = artifacts['target_idx']
                self.prediction_steps = artifacts.get('prediction_steps', 1)
                
                # Recria a arquitetura e carrega os pesos
                model = StockLSTM(input_dim=self.num_features)
                model.load_state_dict(torch.load(model_path))
                
                if torch.cuda.is_available():
                    model.cuda()
                else:
                    model.eval()
                
                self.current_model = model
                self.current_model_name = model_name
                return True
            except Exception as e:
                print(f"Erro ao carregar modelo: {e}")
                return False
        return False

    def train(self, request):
        """
        Executa o pipeline completo de treinamento.

        1. Configura o MLflow Logger.
        2. Prepara os dados.
        3. Treina o modelo usando PyTorch Lightning.
        4. Salva os artefatos (.pth e .pkl) para uso futuro.

        Args:
            request (TrainRequest): Objeto contendo hiperparâmetros e configurações.

        Returns:
            str: O ID da execução (Run ID) no MLflow.
        """
        # Suporte híbrido: Docker (env var) vs Local (localhost)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlf_logger = MLFlowLogger(experiment_name="TechChallenge_Training", tracking_uri=tracking_uri)
        mlf_logger.log_hyperparams(request.dict())

        train_loader, val_loader, scaler, num_features, target_idx = get_dataloaders(
            request.symbol, request.start_date, request.end_date, 
            batch_size=request.batch_size, prediction_steps=request.prediction_steps
        )

        # Atualiza estado interno com os metadados do novo treino
        self.scaler = scaler
        self.num_features = num_features
        self.target_idx = target_idx
        self.prediction_steps = request.prediction_steps
        self.current_model_name = request.model_name

        model = StockLSTM(input_dim=num_features)
        
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1 if torch.cuda.is_available() else "auto"

        trainer = pl.Trainer(
            max_epochs=request.epochs,
            logger=mlf_logger,
            accelerator=accelerator,
            devices=devices,
            callbacks=[PerformanceMonitorCallback()],
            default_root_dir="models/checkpoints" # Evita poluição na raiz
        )

        trainer.fit(model, train_loader, val_loader)
        self.current_model = model

        # Persistência dos artefatos
        torch.save(model.state_dict(), f"models/{request.model_name}.pth")
        joblib.dump({
            'scaler': scaler,
            'num_features': num_features,
            'target_idx': target_idx,
            'prediction_steps': request.prediction_steps
        }, f"models/{request.model_name}.pkl")

        return trainer.logger.run_id

# Instância global compartilhada
model_service = ModelService()