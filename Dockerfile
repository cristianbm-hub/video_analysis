FROM python:3.9-slim

# Crear y establecer el directorio de trabajo
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
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY app.py /app/
COPY mis_funciones/ /app/mis_funciones/

# Verificar que app.py existe
RUN ls -la /app/app.py || exit 1

# Configurar el PYTHONPATH y asegurar permisos
ENV PYTHONPATH=/app
RUN chmod 755 /app/app.py

# Exponer el puerto que usa la aplicación
EXPOSE 5000

# Comando para ejecutar la aplicación con Gunicorn
CMD ["gunicorn", "--log-level", "debug", "--chdir", "/app", "--bind", "0.0.0.0:5000", "app:app"]
