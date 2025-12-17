# 📈 Tech Challenge - Fase 4: Stock Prediction API

Este projeto consiste em uma solução End-to-End de Machine Learning Engineering desenvolvida para o Tech Challenge da Fase 4 (Pós-Tech FIAP).

O objetivo é prever preços de fechamento de ações utilizando redes neurais LSTM (Long Short-Term Memory), servidas por uma API RESTful modularizada, conteinerizada e monitorada.

---

## 🛠️ Tecnologias Utilizadas

![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)

---

## 🚀 Funcionalidades principais

- **Deep Learning com PyTorch Lightning:** implementação de rede LSTM otimizada para séries temporais.
- **API Inteligente (Stateful):** O endpoint de predição carrega automaticamente as configurações usadas no treino (símbolo, janela temporal e features), evitando erros manuais.
- **Experiment Tracking (MLflow):** rastreio completo de métricas (RMSE, MAE, R²), hiperparâmetros e artefatos.
- **Monitoramento de hardware:** hooks personalizados para monitorar uso de CPU, RAM e GPU (VRAM) durante treino e inferência.
- **Arquitetura híbrida:** suporte transparente para execução em Docker (CPU/produção) e local (GPU/desenvolvimento).
- **Prevenção de Data Leakage:** pipeline de dados com normalização ajustada apenas no conjunto de treino.

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
* **Horizonte Flexível (1 a N dias):** O modelo suporta treinamento dinâmico para diferentes horizontes de previsão. Através do parâmetro `prediction_steps`, é possível treinar redes especializadas em prever o dia seguinte (D+1), a próxima semana (D+7) ou qualquer intervalo arbitrário (D+N).

### 2. Camada de Aplicação (Pasta `app/`)
A API foi construída sobre o **FastAPI** pela sua natureza assíncrona e validação automática de tipos (Pydantic).
* **Padrão Singleton:** A classe `ModelService` (`app/services.py`) implementa o padrão Singleton para manter o modelo carregado em memória. Isso evita o custo de I/O a cada requisição, garantindo latência de inferência na ordem de milissegundos.
* **Inferência Inteligente:** A API gerencia o estado dos modelos. Ao carregar um modelo treinado, ela recupera automaticamente o *lookback* (tamanho da janela) e as *features* exatas usadas no treinamento, garantindo que a entrada da predição seja sempre compatível.

### 3. Infraestrutura Híbrida
A solução suporta dois modos de execução sem alteração de código, graças à gestão dinâmica de variáveis de ambiente:
* **Ambiente Docker (Produção):** Focado em estabilidade e portabilidade (CPU). O banco de dados do MLflow é persistido em volume Docker.
* **Ambiente Local (Desenvolvimento):** Focado em performance de treino, permitindo o uso direto de **GPUs NVIDIA** (via CUDA) para acelerar o aprendizado profundo.

---

## 📊 Fonte de Dados

O sistema consome dados históricos do mercado financeiro em tempo real, garantindo que o modelo seja treinado com informações atualizadas.

- **Provedor:** Yahoo Finance (via biblioteca `yfinance`).
- **Flexibilidade:** A API aceita qualquer *ticker* de ação listado na bolsa (ex: `DIS`, `AAPL`, `PETR4.SA`, `^BVSP`).
- **Coleta Sob Demanda:** Os dados não são estáticos; eles são baixados dinamicamente no momento do treino (`POST /train`) com base no intervalo de datas (`start_date`, `end_date`) fornecido pelo usuário.

Para fins de validação do desafio, foram realizados testes utilizando ações do ticker DIS (Disney).


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
* **Solução:** O arquivo `docker-compose.yml` já trata isso mapeando a pasta `/mlflow_data`. Se persistir, apague a pasta `mlflow_data` local e reinicie o Docker.

### 3. GPU não detectada (Execução Local)
Se o log mostrar `CUDA available: False` mesmo você tendo uma placa NVIDIA.
* **Solução:** Verifique se instalou o [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) compatível com seu PyTorch. O projeto funcionará normalmente em CPU (apenas mais lento).

---

## 📡 Utilizando a API

Acesse a documentação interativa (Swagger UI): http://localhost:8000/docs

### 1. Treinar um modelo (`POST /train`)
Nesta etapa, você define a arquitetura temporal (lookback) e o alvo da previsão.

Exemplo de payload:

```json
{
  "model_name": "disney_v1",
  "symbol": "DIS",
  "start_date": "2018-01-01",
  "end_date": "2025-10-30",
  "epochs": 5,
  "batch_size": 32,
  "prediction_steps": 1,
  "lookback_days": 60
}
```

2. Fazer uma previsão (POST /predict)  
   Exemplo de payload:

```json
{
  "model_name": "disney_v1"
}
```

---

### 📘 Detalhamento dos Parâmetros

#### 1. Treinamento (`POST /train`)
| Parâmetro | Tipo | Descrição |
| :--- | :--- | :--- |
| `model_name` | `string` | Identificador único para salvar o modelo (ex: "v1_disney"). |
| `symbol` | `string` | Ticker da ação no Yahoo Finance (ex: "DIS", "AAPL", "PETR4.SA"). |
| `start_date` | `yyyy-mm-dd` | Início do período histórico de dados para treino. |
| `end_date` | `yyyy-mm-dd` | Fim do período histórico. |
| `epochs` | `int` | Ciclos completos de treinamento sobre o dataset. |
| `batch_size` | `int` | Quantidade de dados processados por lote. |
| `prediction_steps` | `int` | **Horizonte:** Quantos dias à frente queremos prever (1=amanhã, 7=semana que vem). |
| `lookback_days` | `int` | **Janela de Memória:** Quantos dias passados a LSTM analisará para tomar decisão. |

#### 2. Predição (`POST /predict`)
| Parâmetro | Tipo | Descrição |
| :--- | :--- | :--- |
| `model_name` | `string` | Nome do modelo previamente treinado a ser carregado. |

---

## 📊 Monitoramento e métricas

Acesse o dashboard do MLflow: http://localhost:5000

O sistema registra automaticamente:
- Métricas de negócio: preço previsto vs real.
- Métricas de modelo: loss, MAE, RMSE, MAPE, R².
- Infraestrutura: consumo de RAM (MB), uso de CPU (%) e GPU VRAM (se disponível).
- Latência: tempo de resposta da inferência.

---

### 📈 Resultados Obtidos (Epoch 199)

Abaixo, apresentamos a convergência do modelo durante um treinamento de 199 épocas. O modelo final atingiu estabilidade com métricas competitivas para séries temporais financeiras.

| Convergência (Loss) | Qualidade do Ajuste (R² Score) |
|:---:|:---:|
| ![Loss Graph](docs/img/val_loss.png) | ![R2 Graph](docs/img/val_r2.png) |
| *A curva de perda (MSE) estabiliza rapidamente, indicando aprendizado efetivo sem underfitting severo.* | *O R² próximo de 0.77 demonstra que o modelo consegue explicar a maior parte da variância dos preços.* |

#### 📊 Métricas Finais (Validação):

| Métrica | Valor Final | Significado |
| :--- | :--- | :--- |
| **Val Loss (MSE)** | `0.000196` | Erro Quadrático Médio. A função de custo minimizada pelo modelo. |
| **MAE** | `0.00942` | *Mean Absolute Error*. O erro médio absoluto em dólares/reais (escala normalizada). |
| **MAPE** | `0.0601` | *Mean Absolute Percentage Error*. A porcentagem média de erro por predição. |
| **RMSE** | `0.013` | *Root Mean Squared Error*. Penaliza erros maiores mais severamente que o MAE. |
| **R² Score** | `0.771` | *Coefficient of Determination*. Indica quão bem o modelo se ajusta aos dados (1.0 é perfeito). |

*(Valores referentes ao último checkpoint salvo na época 199)*

---

## 🧠 Detalhes técnicos

### Prevenção de Data Leakage
Um erro comum em séries temporais é normalizar o dataset inteiro antes da divisão. Neste projeto, o MinMaxScaler é ajustado (fit) apenas nos dados de treino (primeiros 80%) e aplicado (transform) nos dados de validação. Assim, o modelo não tem acesso a estatísticas do futuro.

### Persistência de Metadados e Consistência
Para garantir a robustez da API, não salvamos apenas os pesos da rede neural (`.pth`). Salvamos também um arquivo de artefatos (`.pkl`) contendo:
- O **Scaler** ajustado (para desnormalizar a saída corretamente).
- A lista de **Features** usadas (para garantir que a API baixe as colunas corretas, ex: Open, Close, Volume).
- Os hiperparâmetros estruturais (**Lookback** e **Horizonte**).

Isso permite que a API evolua (ex: adicionando novos indicadores técnicos no futuro) sem quebrar a compatibilidade com modelos antigos.
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

---

## ⚠️ Disclaimer

Este projeto tem fins estritamente educacionais para demonstração de conhecimentos em Engenharia de Machine Learning. As previsões geradas pelo modelo **não constituem recomendação de investimento**. O mercado financeiro é volátil e modelos baseados puramente em preços passados podem não capturar eventos exógenos.