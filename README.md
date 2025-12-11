# 📈 Tech Challenge - Fase 4: Stock Prediction API

Este projeto consiste em uma solução End-to-End de Machine Learning Engineering desenvolvida para o Tech Challenge da Fase 4 (Pós-Tech FIAP).

O objetivo é prever preços de fechamento de ações utilizando redes neurais LSTM (Long Short-Term Memory), servidas por uma API RESTful modularizada, conteinerizada e monitorada.

## 🚀 Funcionalidades principais

- Deep Learning com PyTorch Lightning: implementação de rede LSTM otimizada para séries temporais.
- API RESTful (FastAPI): endpoints assíncronos para treinamento e inferência em tempo real.
- Experiment Tracking (MLflow): rastreio completo de métricas (RMSE, MAE, R²), hiperparâmetros e artefatos.
- Monitoramento de hardware: hooks personalizados para monitorar uso de CPU, RAM e GPU (VRAM) durante treino e inferência.
- Arquitetura híbrida: suporte transparente para execução em Docker (CPU/produção) e local (GPU/desenvolvimento).
- Prevenção de Data Leakage: pipeline de dados com normalização ajustada apenas no conjunto de treino.

---

## 📂 Estrutura do projeto

```text
/
├── app/                    # Lógica da aplicação (API)
│   ├── main.py             # Entrypoint da API e rotas
│   ├── services.py         # Orquestrador de treino e inferência (Singleton)
│   ├── schemas.py          # Contratos de dados (Pydantic)
│   └── config.py           # Configurações globais e logs
│
├── ml/                     # Núcleo de Machine Learning
│   ├── model.py            # Arquitetura LSTM (LightningModule)
│   ├── dataset.py          # ETL e pré-processamento (yfinance)
│   └── callbacks.py        # Monitoramento de hardware
│
├── models/                 # Persistência de modelos (.pth e .pkl)
├── mlruns/                 # Logs locais do MLflow (se rodar localmente)
├── Dockerfile              # Definição da imagem da API
├── docker-compose.yml      # Orquestração (API + MLflow + SQLite)
└── requirements.txt        # Dependências do projeto
```

---

## 🛠️ Como executar

### Opção A: Via Docker (recomendado)
Esta opção garante um ambiente isolado e reproduzível. O MLflow e a API subirão automaticamente.

Certifique-se de ter Docker e Docker Compose instalados. Na raiz do projeto, execute:

```bash
docker-compose up --build
```

A seguir, os serviços que serão iniciados:
- API (Swagger): http://localhost:8000/docs
- MLflow UI: http://localhost:5000

### Opção B: Execução local (desenvolvimento/GPU)
Use esta opção se desejar treinar usando uma GPU NVIDIA (CUDA).

Crie e ative um ambiente virtual:

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Inicie a aplicação (como módulo):

```bash
python -m app.main
```

(Nota: ao rodar localmente, o MLflow abrirá uma interface própria em background na porta 5000.)

---

## 📡 Utilizando a API

Acesse a documentação interativa (Swagger UI): http://localhost:8000/docs

1. Treinar um modelo (POST /train)  
   Exemplo de payload:

```json
{
  "model_name": "disney_v1",
  "symbol": "DIS",
  "start_date": "2018-01-01",
  "end_date": "2025-10-30",
  "epochs": 5,
  "batch_size": 32,
  "prediction_steps": 1
}
```

2. Fazer uma previsão (POST /predict)  
   Exemplo de payload:

```json
{
  "model_name": "disney_v1",
  "symbol": "DIS",
  "lookback_days": 60
}
```

---

## 📊 Monitoramento e métricas

Acesse o dashboard do MLflow: http://localhost:5000

O sistema registra automaticamente:
- Métricas de negócio: preço previsto vs real.
- Métricas de modelo: loss, MAE, RMSE, MAPE, R².
- Infraestrutura: consumo de RAM (MB), uso de CPU (%) e GPU VRAM (se disponível).
- Latência: tempo de resposta da inferência.

---

## 🧠 Detalhes técnicos

### Prevenção de Data Leakage
Um erro comum em séries temporais é normalizar o dataset inteiro antes da divisão. Neste projeto, o MinMaxScaler é ajustado (fit) apenas nos dados de treino (primeiros 80%) e aplicado (transform) nos dados de validação. Assim, o modelo não tem acesso a estatísticas do futuro.

### Persistência robusta
Ao salvar um modelo, geramos dois arquivos na pasta models/:

- {nome}.pth: pesos da rede neural (state dict).
- {nome}.pkl: metadados (scaler ajustado, número de features, horizonte de previsão), necessários para a desnormalização na inferência.

---

## 📝 Autores
Desenvolvido para o Tech Challenge Fase 4 - Pós-Tech Machine Learning Engineering.