"""
Definição da arquitetura da Rede Neural.

Este módulo contém a classe LightningModule que define a estrutura da LSTM,
a função de perda e as métricas monitoradas durante o treinamento.
"""
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
import torchmetrics

class StockLSTM(pl.LightningModule):
    """
    Modelo LSTM para previsão de preços de ações.

    Herda de pl.LightningModule para facilitar a integração com PyTorch Lightning
    e MLflow.

    Attributes:
        lstm (nn.LSTM): Camada recorrente LSTM.
        fc (nn.Linear): Camada totalmente conectada para saída final.
        loss_fn (nn.MSELoss): Função de perda (Mean Squared Error).
        mae (MeanAbsoluteError): Métrica de erro absoluto médio.
        rmse (MeanSquaredError): Métrica de erro quadrático médio.
        mape (MeanAbsolutePercentageError): Métrica de erro percentual.
        r2 (R2Score): Coeficiente de determinação.
    """
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        # Arquitetura
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
         
        # Função de Custo
        self.loss_fn = nn.MSELoss()

        # Métricas
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.r2 = torchmetrics.R2Score()

    def forward(self, x):
        """
        Passagem direta (Forward pass) dos dados pela rede.

        Args:
            x (torch.Tensor): Tensor de entrada com shape (batch, seq_len, features).

        Returns:
            torch.Tensor: Previsão do modelo.
        """
        out, _ = self.lstm(x)
        # Seleciona apenas a saída do último passo temporal
        out = out[:, -1, :]
        prediction = self.fc(out)
        return prediction

    def training_step(self, batch, batch_idx):
        """Executa um passo de treinamento e registra o loss."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Executa um passo de validação e registra múltiplas métricas."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Cálculo de métricas
        mae_val = self.mae(y_hat, y)
        rmse_val = self.rmse(y_hat, y)
        mape_val = self.mape(y_hat, y)
        r2_val = self.r2(y_hat, y)

        # Log
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_mae", mae_val, prog_bar=True, logger=True)
        self.log("val_rmse", rmse_val, prog_bar=True, logger=True)
        self.log("val_mape", mape_val, prog_bar=True, logger=True)
        self.log("val_r2", r2_val, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        """Configura o otimizador Adam."""
        return Adam(self.parameters(), lr=self.hparams.learning_rate)