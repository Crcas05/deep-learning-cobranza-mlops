FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 👇 AQUI CAMBIO IMPORTANTE
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .

CMD ["python", "-u", "src/train_model.py"]