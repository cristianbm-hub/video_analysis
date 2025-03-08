FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y procesamiento de video
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip antes de instalar dependencias
RUN pip install --upgrade pip

# Copiar los archivos de requisitos primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Configurar el PYTHONPATH
ENV PYTHONPATH=/app

# Exponer el puerto que usa la aplicación
EXPOSE 5000

# Comando para ejecutar la aplicación con Gunicorn
CMD ["gunicorn", "--chdir", "/app", "--bind", "0.0.0.0:5000", "app:app"]
