from flask import Flask, request, jsonify
import os
import uuid
import tempfile
import requests
from urllib.parse import urlparse
import cv2
import numpy as np
from supabase import create_client, Client
from datetime import datetime

# Configuración de Supabase
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Importar tus funciones de análisis de video aquí
# from video_analyzer import analyze_video_quality

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_filename_from_url(url):
    """Extrae el nombre del archivo de una URL"""
    parsed_url = urlparse(url)
    path = parsed_url.path
    filename = os.path.basename(path)
    
    # Si no hay nombre de archivo en la URL, generar uno
    if not filename or '.' not in filename:
        return f"video_{uuid.uuid4()}.mp4"
    
    return filename

def download_video(url):
    """Descarga un video desde una URL y devuelve la ruta local"""
    try:
        # Obtener el nombre del archivo de la URL
        filename = get_filename_from_url(url)
        
        # Generar un nombre único para evitar colisiones
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Descargar el archivo
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Lanzar excepción si hay error HTTP
        
        # Guardar el archivo
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return filepath
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")

# Función de ejemplo para análisis de video
# Reemplaza esto con tus propias funciones de análisis
def analyze_video_quality(video_path):
    """
    Función de ejemplo que simula el análisis de calidad de video.
    Reemplaza esta función con tu implementación real.
    
    """
    
    """Analiza la calidad del video en términos de brillo, contraste y ruido y devuelve un JSON."""
    cap = cv2.VideoCapture(video_path)
    brightness_vals = []
    contrast_vals = []
    noise_vals = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        brightness_vals.append(brightness)

        contrast = np.std(gray)
        contrast_vals.append(contrast)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_vals.append(laplacian)

    cap.release()

    #retorno test (BORRAR En PRO)
    return {
        "score": {
            "audio": 0.78,
            "visual": 0.82,
            "overall": 0.85,
            "technical": 0.92,
            "engagement": 0.88
        },
        "insights": {
            "bestMoments": [
            {
                "reason": "Alta interacción durante la demostración práctica del método Pomodoro",
                "timestamp": 127.5
            },
            {
                "reason": "Pico de engagement durante la revelación del tip #1",
                "timestamp": 245.8
            },
            {
                "reason": "Momento memorable durante la conclusión con call-to-action",
                "timestamp": 315.2
            }
            ],
            "contentSummary": "Video educativo sobre productividad que presenta 5 técnicas principales: método Pomodoro, planificación semanal, gestión de energía, eliminación de distracciones y rutinas matutinas. Incluye demostraciones prácticas y casos de estudio.",
            "viralPotential": 0.78,
            "audienceRetention": [
            {
                "score": 1,
                "timestamp": 0
            },
            {
                "score": 0.95,
                "timestamp": 60
            },
            {
                "score": 0.92,
                "timestamp": 120
            },
            {
                "score": 0.88,
                "timestamp": 180
            },
            {
                "score": 0.85,
                "timestamp": 240
            },
            {
                "score": 0.82,
                "timestamp": 300
            }
            ]
        },
        "technical": {
            "bitrate": 8500000,
            "frameRate": 30,
            "resolution": "1920x1080",
            "audioQuality": {
            "noise": 0.12,
            "volume": 0.75,
            "clarity": 0.85
            }
        },
        "recommendations": {
            "tags": [
            "productividad",
            "desarrollo personal",
            "tips",
            "trabajo",
            "organización",
            "gestión del tiempo"
            ],
            "title": "5 Tips Infalibles para Mejorar tu Productividad | Guía Definitiva 2024",
            "description": "Descubre las técnicas más efectivas para aumentar tu productividad diaria con estos 5 consejos probados científicamente. Incluye ejemplos prácticos y herramientas recomendadas.",
            "improvements": [
            {
                "category": "audio",
                "priority": "high",
                "suggestion": "Reducir el ruido de fondo en el segmento 2:15-3:45"
            },
            {
                "category": "visual",
                "priority": "medium",
                "suggestion": "Aumentar la iluminación en las tomas de la oficina"
            },
            {
                "category": "content",
                "priority": "low",
                "suggestion": "Añadir ejemplos prácticos para el tercer consejo"
            },
            {
                "category": "technical",
                "priority": "medium",
                "suggestion": "Estabilizar el video durante las tomas en movimiento"
            }
            ],
            "thumbnailTimestamps": [
            45.5,
            127.8,
            198.2,
            245.6
            ]
        }
        }
"""
    return {
        "brightness_mean": round(np.mean(brightness_vals), 2),
        "contrast_mean": round(np.mean(contrast_vals), 2),
        "noise_mean": round(np.mean(noise_vals), 2)
    }
"""


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/analyze', methods=['POST'])
def analyze_video():
    data = request.json
    
    # Verificar si hay una URL en la solicitud
    if not data or 'video_url' not in data:
        return jsonify({"error": "No video URL provided"}), 400
    
    video_url = data['video_url']
    
    try:
        # Descargar el video
        filepath = download_video(video_url)
        
        try:
            # Analizar el video
            results = analyze_video_quality(filepath)
            
            # Actualizar el registro existente en Supabase
            try:
                # Buscar el video por su URL original
                data, error = supabase.table('videos') \
                    .update({"analysis_result": results}) \
                    .eq('original_url', video_url) \
                    .execute()
                
                if error:
                    print(f"Error al actualizar en Supabase: {error}")
                elif not data[1] or len(data[1]) == 0:
                    print(f"No se encontró el video con URL: {video_url}")
            except Exception as e:
                print(f"Error al interactuar con Supabase: {str(e)}")
            
            # Limpiar el archivo temporal
            os.remove(filepath)
            
            return jsonify({
                "success": True,
                "video_url": video_url,
                "analysis": results
            })
        except Exception as e:
            # Limpiar el archivo temporal en caso de error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"Analysis error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
