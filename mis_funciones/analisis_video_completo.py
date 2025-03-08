# 📌 1. INSTALAR DEPENDENCIAS
!pip install ffmpeg-python opencv-python ultralytics google-cloud-videointelligence
!apt-get install -y ffmpeg
!apt-get install -y tesseract-ocr
!pip install pytesseract

import requests
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.cluster import KMeans
from google.cloud import videointelligence
import pytesseract
import json


# 📌 2. DESCARGAR VIDEO DESDE URL
def descargar_video(url, nombre_archivo):
    """ Descarga un video desde una URL y lo guarda en el sistema local de Google Colab. """
    respuesta = requests.get(url, stream=True)
    if respuesta.status_code == 200:
        with open(nombre_archivo, 'wb') as f:
            for chunk in respuesta.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"✅ Video descargado: {nombre_archivo}")
        return nombre_archivo
    else:
        print("❌ Error al descargar el video")
        return None

# URL del video (Cambia esto por tu URL real)
url_video = "https://ezfzuensxdnfsfgttzjt.supabase.co/storage/v1/object/public/videos/33ce77bb-864b-4a56-8094-1fc751423088/1741395990129-2785533-sd_360_640_25fps.mp4"

# Nombre del archivo descargado
video_local = "video_descargado.mp4"

# Descargar el video
descargar_video(url_video, video_local)

# 📌 3. REDUCIR FPS Y ACELERAR EL VIDEO
def reducir_fps_y_acelerar(video_entrada, video_salida, fps_deseado, velocidad):
    """ Reduce los FPS y acelera la velocidad del video. """
    cap = cv2.VideoCapture(video_entrada)
    fps_original = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    nuevo_fps = min(fps_deseado, fps_original)
    out = cv2.VideoWriter(video_salida, fourcc, nuevo_fps, (width, height))

    frame_count = 0
    frame_interval = max(1, int(fps_original / nuevo_fps))
    velocidad_intervalo = max(1, int(velocidad))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (frame_interval * velocidad_intervalo) == 0:
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()
    print(f"✅ Video reducido guardado en: {video_salida}")

# Reducir FPS y acelerar el video
video_reducido = "video_reducido.mp4"
reducir_fps_y_acelerar(video_local, video_reducido, fps_deseado=5, velocidad=5)

# 📌 4. ANALIZAR CALIDAD DEL VIDEO (BRILLO, CONTRASTE, RUIDO)
def analizar_calidad(video_path):
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

    result = {
        "brightness_mean": round(np.mean(brightness_vals), 2),
        "contrast_mean": round(np.mean(contrast_vals), 2),
        "noise_mean": round(np.mean(noise_vals), 2)
    }

    print(json.dumps(result, indent=4))
    return json.dumps(result, indent=4)

# Ejecutar análisis de calidad
analizar_calidad(video_reducido)

# 📌 5. ANALIZAR ENCUADRE CON YOLOv8
modelo = YOLO("yolov8n.pt")

def analizar_encuadre(video_path):
    """ Evalúa si el sujeto principal está centrado en el video y devuelve los datos en formato JSON. """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detecciones = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 20 == 0:
            resultados = modelo(frame)

            for box in resultados[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                detecciones.append({
                    "frame": frame_count,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
        
        frame_count += 1

    cap.release()
    print(json.dumps(detecciones, indent=4))
    return json.dumps(detecciones, indent=4)

# Ejecutar análisis de encuadre
analizar_encuadre(video_reducido)


#paleta de colores
def extraer_colores(video_path, num_colores=5, sample_rate=10):
    """ Extrae la paleta de colores predominantes de un video usando K-means clustering. """

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Tomar una muestra de frames para análisis
        if frame_count % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB
            frames.append(frame.reshape(-1, 3))  # Convertir en una lista de píxeles

        frame_count += 1

    cap.release()

    # Combinar los píxeles de todos los frames tomados
    pixels = np.vstack(frames)

    # Aplicar K-Means Clustering para encontrar los colores predominantes
    kmeans = KMeans(n_clusters=num_colores, random_state=0, n_init=10)
    kmeans.fit(pixels)
    colores = kmeans.cluster_centers_.astype(int)  # Convertir a enteros

    # Crear una imagen con la paleta de colores
    paleta = np.zeros((50, 300, 3), dtype=np.uint8)
    step = 300 // num_colores

    for i, color in enumerate(colores):
        paleta[:, i * step:(i + 1) * step] = color  # Pintar cada bloque de color

    # Mostrar la paleta de colores
    plt.figure(figsize=(8, 2))
    plt.imshow(paleta)
    plt.axis("off")
    plt.title("Paleta de Colores Predominantes")
    plt.show()

    # Devolver los colores extraídos
    resultado = [f"Color {i+1}: {color}" for i, color in enumerate(colores)]
    print(json.dumps(resultado, indent=4))
    return json.dumps(resultado, indent=4)


# Ejecutar la extracción de colores en el video
colores_predominantes = extraer_colores(video_reducido, num_colores=5, sample_rate=20)




#Cambios de escena

def analizar_cortes_escena(video_path, umbral=0.5):
    """ Detecta cortes de escena y mide el ritmo del video. """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS del video
    cambios = 0
    duraciones = []
    frame_count = 0
    last_cut = 0

    ret, prev_frame = cap.read()
    if not ret:
        return json.dumps({"error": "No se pudo leer el video."})

    prev_hist = cv2.calcHist([cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)], [0], None, [256], [0, 256])
    prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Calcular histograma del frame actual
        hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Comparar histograma con el frame anterior
        correlacion = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

        # Si la diferencia es grande, detectamos un cambio de toma
        if correlacion < umbral:
            cambios += 1
            tiempo = frame_count / fps
            duraciones.append(tiempo - last_cut)
            last_cut = tiempo

        prev_hist = hist

    cap.release()

    # Calcular la duración promedio de las tomas
    duracion_promedio = np.mean(duraciones) if duraciones else 0
    duracion_promedio *= 5  # Multiplica la duración promedio de cada toma por 5

    resultado = {
        "cortes_detectados": cambios,
        "duracion_promedio": round(duracion_promedio, 2),
        "ritmo": "dinamico" if cambios > 10 and duracion_promedio < 3 else "lento-estatico"
    }

    print(json.dumps(resultado, indent=4))
    return json.dumps(resultado, indent=4)

# Ejecutar análisis
analizar_cortes_escena(video_reducido)

def analizar_movimiento(video_path, umbral=2.0):
    """ Analiza el movimiento del video para detectar estabilidad o dinamismo y devuelve un JSON con el resultado """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return json.dumps({"error": "No se pudo leer el video."})

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_movimiento = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow para medir desplazamiento entre frames
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        movimiento = float(np.mean(np.abs(flow)))  # Conversión a float nativo
        total_movimiento.append(movimiento)

        prev_gray = gray

    cap.release()

    promedio_movimiento = float(np.mean(total_movimiento)) if total_movimiento else 0.0  # Conversión a float nativo

    resultado = {
        "nivel_movimiento": round(promedio_movimiento, 2),
        "es_dinamico": promedio_movimiento >= umbral,
        "mensaje": "El video tiene suficiente dinamismo." if promedio_movimiento >= umbral else "El video es estático o poco dinámico."
    }

    print(json.dumps(resultado, ensure_ascii=False, indent=4))
    return json.dumps(resultado, ensure_ascii=False, indent=4)

# Ejecutar análisis de movimiento
analizar_movimiento(video_reducido)



def detectar_rostros(video_path):
    """Detecta rostros en el video para evaluar si están bien encuadrados y devuelve un resultado en JSON."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    rostros_detectados = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(rostros) > 0:
            rostros_detectados += 1

        frame_count += 1

    cap.release()

    porcentaje_rostros = (rostros_detectados / frame_count) * 100 if frame_count > 0 else 0
    resultado = {
        "porcentaje_rostros": round(porcentaje_rostros, 2),
        "presencia_personas": porcentaje_rostros > 50
    }

    print(json.dumps(resultado, ensure_ascii=False))
    return json.dumps(resultado, ensure_ascii=False)


# Ejecutar detección de rostros
detectar_rostros(video_reducido)


def detectar_texto(video_path, sample_rate=10):
    """Detecta texto en pantalla en varios frames del video y devuelve un JSON"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    textos_detectados = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            texto = pytesseract.image_to_string(frame, lang="eng")
            if texto.strip():
                textos_detectados.append(texto.strip())

        frame_count += 1

    cap.release()
    print(json.dumps({"textos_detectados": textos_detectados}, ensure_ascii=False, indent=4))
    return json.dumps({"textos_detectados": textos_detectados}, ensure_ascii=False, indent=4)


# Ejecutar detección de texto en pantalla
detectar_texto(video_reducido)




# 📌 7. (OPCIONAL) BORRAR ARCHIVOS TEMPORALES PARA LIBERAR ESPACIO
os.remove(video_local)
os.remove(video_reducido)

print("✅ Todo listo. ¡Análisis completado! 🚀")
