# 📌 INSTALAR DEPENDENCIAS
!pip install openai-whisper ffmpeg-python requests
!apt-get install -y ffmpeg  # Asegurar que FFmpeg está instalado

import whisper
import requests
import subprocess
import os
from google.colab import files

# 📌 1. DESCARGAR VIDEO DESDE URL
def descargar_video(url, nombre_archivo):
    """ Descarga un video desde una URL y lo guarda localmente. """
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

# URL del video (cambia esto por la URL real del video)
url_video = "https://ezfzuensxdnfsfgttzjt.supabase.co/storage/v1/object/public/videos/33ce77bb-864b-4a56-8094-1fc751423088/1741390491132-2025-02-19%2020-44-34.mp4"

# Nombre del archivo descargado
video_local = "video_descargado.mp4"

# Descargar el video
descargar_video(url_video, video_local)

# 📌 2. EXTRAER AUDIO DEL VIDEO
def extraer_audio(video_path, audio_path):
    """ Extrae el audio de un video y lo guarda en formato WAV. """
    comando = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path} -y"
    subprocess.run(comando, shell=True)
    print(f"✅ Audio extraído: {audio_path}")

# Archivo de salida de audio
audio_local = "audio_extraido.wav"

# Extraer audio
extraer_audio(video_local, audio_local)

# 📌 3. TRANSCRIBIR AUDIO CON WHISPER
def transcribir_audio(audio_path, model_size="small"):
    """ Usa Whisper para transcribir el audio. """
    modelo = whisper.load_model(model_size)  # Modelos: tiny, base, small, medium, large
    resultado = modelo.transcribe(audio_path)
    return resultado["text"]

# Transcribir el audio
texto_transcrito = transcribir_audio(audio_local)

# Mostrar la transcripción
print("\n📝 Transcripción del video:\n")
print(texto_transcrito)

# 📌 4. GUARDAR TRANSCRIPCIÓN EN UN ARCHIVO .TXT
with open("transcripcion.txt", "w") as f:
    f.write(texto_transcrito)

print("✅ Transcripción guardada en transcripcion.txt")

# Descargar el archivo de transcripción
files.download("transcripcion.txt")

# 📌 5. GUARDAR TRANSCRIPCIÓN COMO SUBTÍTULOS .SRT
def guardar_como_srt(texto, archivo_srt):
    lineas = texto.split(". ")
    with open(archivo_srt, "w") as f:
        for i, linea in enumerate(lineas, 1):
            f.write(f"{i}\n00:00:0{i},000 --> 00:00:0{i+1},000\n{linea}\n\n")

guardar_como_srt(texto_transcrito, "subtitulos.srt")

# Descargar el archivo .srt
files.download("subtitulos.srt")

# 📌 6. (OPCIONAL) BORRAR ARCHIVOS TEMPORALES PARA LIBERAR ESPACIO
os.remove(video_local)
os.remove(audio_local)

print("✅ Todo listo. ¡Transcripción completada! 🚀")
