# Usar una imagen base oficial de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY src/ ./src/
COPY data/ ./data/

# Crear directorios necesarios
RUN mkdir -p test/embeddings

# Establecer el directorio de trabajo en src
WORKDIR /app/src

# Exponer puerto para FastAPI
EXPOSE 8000

# Comando por defecto - ejecutar FastAPI
CMD ["python", "EmbeddingGenerator/main.py"]
