import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
import torchmetrics

class StockLSTM(pl.LightningModule):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=1, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        # Arquitetura LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
         
        # Função de Perda
        self.loss_fn = nn.MSELoss()

        # Métricas de Avaliação
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.mape = torchmetrics.MeanAbsolutePercentageError()
        self.r2 = torchmetrics.R2Score()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Pega apenas o último estado oculto
        out = out[:, -1, :]
        prediction = self.fc(out)
        return prediction

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Calcular métricas
        mae_val = self.mae(y_hat, y)
        rmse_val = self.rmse(y_hat, y)
        mape_val = self.mape(y_hat, y)
        r2_val = self.r2(y_hat, y)

        # Logar métricas
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_mae", mae_val, prog_bar=True, logger=True)
        self.log("val_rmse", rmse_val, prog_bar=True, logger=True)
        self.log("val_mape", mape_val, prog_bar=True, logger=True)
        self.log("val_r2", r2_val, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)