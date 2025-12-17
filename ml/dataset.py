"""
Módulo de processamento de dados e datasets.

Este módulo é responsável por baixar dados financeiros, realizar o pré-processamento
(normalização e janelamento) e fornecer os DataLoaders para o PyTorch.
Define também as features globais utilizadas pelo modelo.
"""
import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Define as colunas usadas globalmente para garantir consistência entre treino e inferência
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']

class StockDataset(Dataset):
    """
    Dataset personalizado para séries temporais financeiras.

    Args:
        sequences (torch.FloatTensor): Tensor contendo as sequências de entrada (features).
        targets (torch.FloatTensor): Tensor contendo os valores alvo (targets).
    """
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def download_and_process_data(symbol, start_date, end_date, seq_length=60, prediction_steps=1):
    """
    Baixa dados do Yahoo Finance e prepara as sequências para treino/teste.

    Realiza a normalização dos dados garantindo que não haja vazamento de dados (Data Leakage),
    ajustando o scaler apenas nos dados de treino. Utiliza a constante global FEATURES.

    Args:
        symbol (str): O ticker da ação (ex: 'DIS', 'AAPL').
        start_date (str): Data de início no formato 'YYYY-MM-DD'.
        end_date (str): Data de fim no formato 'YYYY-MM-DD'.
        seq_length (int): Tamanho da janela de tempo (lookback) para entrada do modelo.
        prediction_steps (int): Horizonte de previsão (quantos dias à frente).

    Returns:
        tuple: Contendo (X, y, scaler, target_col_index).
    
    Raises:
        ValueError: Se nenhum dado for encontrado para o símbolo especificado.
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {symbol}.")

    data = df[FEATURES].dropna().values 
    
    # Prevenção de Data Leakage: Fit apenas no conjunto de treino (primeiros 80%)
    train_size_raw = int(len(data) * 0.8)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_size_raw])
    
    # Aplica a transformação em todos os dados usando a escala aprendida no treino
    scaled_data = scaler.transform(data)
    
    target_col_index = FEATURES.index('Close')
    
    X, y = [], []
    
    # Ajuste do limite para garantir que target_idx não ultrapasse o array
    limit = len(scaled_data) - seq_length - prediction_steps + 1
    
    for i in range(limit):
        X.append(scaled_data[i:i+seq_length]) 
        target_idx = i + seq_length + prediction_steps - 1
        y.append(scaled_data[target_idx, target_col_index])
        
    X = np.array(X) 
    y = np.array(y)
    y = y.reshape(-1, 1)

    return X, y, scaler, target_col_index

def get_dataloaders(symbol, start, end, seq_len=60, prediction_steps=1, batch_size=32, num_workers=0):
    """
    Gera os DataLoaders de treino e validação prontos para o PyTorch Lightning.

    Args:
        symbol (str): Ticker da ação.
        start (str): Data de início.
        end (str): Data de fim.
        seq_len (int): Tamanho da janela histórica.
        prediction_steps (int): Dias à frente para prever.
        batch_size (int): Tamanho do lote de dados.
        num_workers (int): Número de subprocessos para carregar dados.

    Returns:
        tuple: (train_loader, val_loader, scaler, num_features, target_idx)
    """
    X, y, scaler, target_idx = download_and_process_data(symbol, start, end, seq_len, prediction_steps)
    
    train_size = int(len(X) * 0.8)
    
    train_dataset = StockDataset(X[:train_size], y[:train_size])
    val_dataset = StockDataset(X[train_size:], y[train_size:])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )
    
    num_features = X.shape[2] 
    
    return train_loader, val_loader, scaler, num_features, target_idx