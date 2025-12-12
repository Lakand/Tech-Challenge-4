# 📈 Tech Challenge - Fase 4: Stock Prediction API

Este projeto consiste em uma solução End-to-End de Machine Learning Engineering desenvolvida para o Tech Challenge da Fase 4 (Pós-Tech FIAP).

O objetivo é prever preços de fechamento de ações utilizando redes neurais LSTM (Long Short-Term Memory), servidas por uma API RESTful modularizada, conteinerizada e monitorada.

## 🛠️ Tecnologias Utilizadas

![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)

---

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
│   ├── utils.py            # Utilitários de Hardware (GPU)
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
├── requirements.txt        # Dependências do projeto
└── .gitignore              # Arquivos ignorados pelo Git
```

---

## 🏗️ Arquitetura da Solução

O projeto foi desenhado seguindo princípios de **Clean Architecture** e **MLOps**, visando a separação clara entre a ciência de dados e a engenharia de software.

### 1. Núcleo de Inteligência (Pasta `ml/`)
Optou-se por uma arquitetura **LSTM (Long Short-Term Memory)** devido à sua capacidade superior de capturar dependências de longo prazo em séries temporais financeiras.
* **Framework:** PyTorch Lightning foi escolhido para abstrair o *loop* de treino, facilitar o uso de GPU e integrar nativamente com o MLflow.
* **Horizonte Flexível (1 a N dias):** O modelo suporta treinamento dinâmico para diferentes horizontes de previsão. Através do parâmetro `prediction_steps`, é possível treinar redes especializadas em prever o dia seguinte (D+1), a próxima semana (D+7) ou qualquer intervalo arbitrário (D+N), ajustando automaticamente o alvo ($y$) durante o processamento dos dados.

### 2. Camada de Aplicação (Pasta `app/`)
A API foi construída sobre o **FastAPI** pela sua natureza assíncrona e validação automática de tipos (Pydantic).
* **Padrão Singleton:** A classe `ModelService` (`app/services.py`) implementa o padrão Singleton para manter o modelo carregado em memória. Isso evita o custo de I/O a cada requisição, garantindo latência de inferência na ordem de milissegundos.
* **Contratos de Dados:** O uso de schemas (`app/schemas.py`) valida rigorosamente as entradas, garantindo que parâmetros críticos como datas e horizontes de previsão estejam no formato correto.

### 3. Infraestrutura Híbrida
A solução suporta dois modos de execução sem alteração de código, graças à gestão dinâmica de variáveis de ambiente:
* **Ambiente Docker (Produção):** Focado em estabilidade e portabilidade (CPU). O banco de dados do MLflow é persistido em volume Docker.
* **Ambiente Local (Desenvolvimento):** Focado em performance de treino, permitindo o uso direto de **GPUs NVIDIA** (via CUDA) para acelerar o aprendizado profundo.

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

## ⚠️ Solução de Problemas Comuns

### 1. Erro: "Port is already allocated"
Se ao rodar o Docker aparecer erro nas portas `8000` ou `5000`, certifique-se de que não há outro serviço rodando (ou uma execução antiga do próprio projeto).
* **Solução:** Pare os containers antigos com `docker-compose down` ou altere o mapeamento no `docker-compose.yml`.

### 2. Erro de Permissão no Banco de Dados (SQLite)
Se o MLflow reclamar de "readonly database" ou "unable to open database file".
* **Solução:** O arquivo `docker-compose.yml` já trata isso mapeando a pasta `/mlflow_data`, mas se persistir, apague a pasta `mlflow_data` local e reinicie o Docker.

### 3. GPU não detectada (Execução Local)
Se o log mostrar `CUDA available: False` mesmo você tendo uma placa NVIDIA.
* **Solução:** Verifique se instalou o [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) compatível com seu PyTorch. O projeto funcionará normalmente em CPU (apenas mais lento).

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

### 📘 Detalhamento dos Parâmetros

Entenda a função de cada campo nas requisições:

#### 1. Treinamento (`POST /train`)
| Parâmetro | Tipo | Descrição |
| :--- | :--- | :--- |
| `model_name` | `string` | Identificador único para salvar o modelo (ex: "v1_disney"). Permite criar múltiplas versões sem sobrescrever. |
| `symbol` | `string` | Ticker da ação no Yahoo Finance (ex: "DIS", "AAPL", "PETR4.SA"). O modelo será treinado neste ativo. |
| `start_date` | `yyyy-mm-dd` | Início do período histórico de dados para treino. |
| `end_date` | `yyyy-mm-dd` | Fim do período histórico. |
| `epochs` | `int` | Número de vezes que o modelo verá o dataset completo. |
| `batch_size` | `int` | Quantidade de dados processados por vez antes de atualizar os pesos. |
| `prediction_steps` | `int` | **Horizonte de Previsão:** Define o alvo da predição. Use `1` para prever o dia seguinte ou `N` para prever o preço daqui a N dias. |

#### 2. Predição (`POST /predict`)
| Parâmetro | Tipo | Descrição |
| :--- | :--- | :--- |
| `model_name` | `string` | Nome do arquivo do modelo (`.pth`) a ser carregado da pasta `models/`. |
| `symbol` | `string` | Ticker do ativo para baixar os dados mais recentes (janela de entrada). |
| `lookback_days` | `int` | **Janela de Contexto:** Quantos dias passados o modelo deve analisar para calcular o futuro. |

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

## 🔮 Próximos Passos e Melhorias Futuras

Para evoluir este projeto em um ambiente produtivo real, as seguintes implementações estão no roadmap:

1.  **Feature Engineering Avançada:** Incluir indicadores técnicos (RSI, MACD, Bandas de Bollinger) além dos preços puros (OHLCV) para enriquecer o contexto do modelo.
2.  **Hyperparameter Tuning:** Implementar [Optuna](https://optuna.org/) para busca automática dos melhores parâmetros da LSTM (learning rate, número de camadas, neurônios).
3.  **Deployment na Nuvem:** Criar pipeline de CI/CD (GitHub Actions) para deploy automático na AWS (ECS ou SageMaker).
4.  **Autenticação na API:** Adicionar camada de segurança (JWT) nos endpoints da FastAPI.

---

## 📝 Autores

Projeto desenvolvido por:
* **Celso Lopes** - RM: 364112 

Desenvolvido para o **Tech Challenge Fase 4** - Pós-Tech Machine Learning Engineering (FIAP).