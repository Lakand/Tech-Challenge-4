import yfinance as yf
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# No arquivo dataset.py

def download_and_process_data(symbol, start_date, end_date, seq_length=60, prediction_steps=1):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {symbol}.")

    data = df[features].dropna().values 
    
    # --- CORREÇÃO DO DATA LEAK ---
    # 1. Definimos onde acaba o treino (80%)
    train_size_raw = int(len(data) * 0.8)
    
    # 2. Criamos o Scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 3. Ajustamos a "régua" (fit) APENAS nos dados de treino
    scaler.fit(data[:train_size_raw])
    
    # 4. Aplicamos a transformação no conjunto todo
    # (Assim o teste é normalizado com a base do passado, simulando a vida real)
    scaled_data = scaler.transform(data)
    # -----------------------------
    
    target_col_index = features.index('Close')
    
    X, y = [], []
    
    limit = len(scaled_data) - seq_length - prediction_steps + 1
    
    for i in range(limit):
        X.append(scaled_data[i:i+seq_length]) 
        target_idx = i + seq_length + prediction_steps - 1
        y.append(scaled_data[target_idx, target_col_index])
        
    X = np.array(X) 
    y = np.array(y)
    y = y.reshape(-1, 1)

    return X, y, scaler, target_col_index

# Atualize também quem chama essa função
def get_dataloaders(symbol, start, end, seq_len=60, prediction_steps=1, batch_size=32, num_workers=0):
    # Passamos o prediction_steps adiante
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