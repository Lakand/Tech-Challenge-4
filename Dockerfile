# 1. Usa uma imagem base leve do Python 3.10
FROM python:3.10-slim

# 2. Define variáveis de ambiente para otimizar o Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Define o diretório de trabalho dentro do container
WORKDIR /app

# 4. Instala dependências do sistema operacional necessárias para compilar pacotes
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Copia o requirements e instala as bibliotecas Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copia todo o código do projeto para dentro do container
# Isso inclui as pastas app/, ml/ e models/
COPY . .

# 7. Expõe a porta 8000 (API)
EXPOSE 8000

# 8. Comando para rodar a API
# Note o "app.main:app", que reflete sua nova estrutura modular
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]