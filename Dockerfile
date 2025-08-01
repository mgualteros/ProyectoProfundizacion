FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos al contenedor
COPY . /app

# Instala dependencias
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download es_core_news_sm \
 && rm -rf /root/.cache

# Comando por defecto
CMD ["python"]
