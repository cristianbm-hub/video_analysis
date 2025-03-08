# API de Análisis de Calidad de Video

Esta API permite analizar la calidad de videos a través de una interfaz REST, recibiendo URLs de videos.

## Requisitos

- Docker y Docker Compose
- Python 3.9 o superior (para pruebas locales)

## Configuración

1. Clona este repositorio
2. Personaliza el archivo `app.py` para integrar tus funciones de análisis de video
3. Actualiza `requirements.txt` con las dependencias específicas que necesites

## Construcción y ejecución con Docker

```bash
# Construir y ejecutar el contenedor
docker-compose up --build

# Ejecutar en segundo plano
docker-compose up -d
```

## Endpoints de la API

### Verificación de salud

```
GET /health
```

Respuesta:
```json
{
  "status": "healthy"
}
```

### Análisis de video por URL

```
POST /analyze
Content-Type: application/json

{
  "video_url": "https://ejemplo.com/ruta/al/video.mp4"
}
```

Respuesta:
```json
{
  "success": true,
  "video_url": "https://ejemplo.com/ruta/al/video.mp4",
  "analysis": {
    "resolution": "1920x1080",
    "bitrate": "8.5 Mbps",
    "framerate": "30 fps",
    "quality_score": 8.7,
    "issues": ["slight compression artifacts at 00:15:30"]
  }
}
```

## Pruebas

Para probar la API localmente:

```bash
python test_api.py https://ejemplo.com/ruta/al/video.mp4
```

## Integración en tu SaaS

Esta API puede ser desplegada como un microservicio independiente y ser consumida por tu aplicación SaaS principal. Puedes:

1. Desplegarla en un servicio de contenedores como AWS ECS, Google Cloud Run o Kubernetes
2. Configurar un balanceador de carga para manejar múltiples instancias
3. Implementar autenticación mediante tokens JWT o API keys
4. Configurar límites de tamaño y tiempos de espera adecuados para videos grandes
