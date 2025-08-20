import cv2
from detection import AccidentDetectionModel
import numpy as np
import os
import winsound
import threading
import requests
from datetime import datetime
import random

# Carregar o modelo
model = AccidentDetectionModel("model.json", 'model_weights.keras')
font = cv2.FONT_HERSHEY_SIMPLEX
API_ENDPOINT = "https://www.seusistemaonline.com/api/report_emergecy.php"

# Debug: ativar mensagens de depuração
DEBUG = True

# Global para detecção de movimento
last_frame_blurred = None

def frame_has_motion(frame, threshold=25, min_motion=3000):
    global last_frame_blurred
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    if last_frame_blurred is None:
        last_frame_blurred = blurred
        return False

    frame_delta = cv2.absdiff(last_frame_blurred, blurred)
    last_frame_blurred = blurred

    thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
    motion_level = np.sum(thresh) / 255

    if DEBUG:
        print(f"[DEBUG] Nível de movimento: {motion_level}")

    return motion_level > min_motion

def send_emergency_report(camera_name, photo_path):
    emergency_id = random.randint(0, 999)
    data_atual = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    payload = {
        "emergency_id": emergency_id,
        "agency_name": "Policia",
        "agency_id": 1337,
        "case_severity": "Critical",
        "emergency_category": "Colisão",
        "dates": data_atual,
        "state": f"Bairro {camera_name}",
        "phone_number": "845694156",
        "address": camera_name,
        "name": f"Colisao {camera_name}",
        "status": "Pendente",
        "email": f"{camera_name.lower().replace(' ', '')}@gmail.com",
        "victim_id": 32535,
        "description": "Colisão entre duas viaturas"
    }

    files = {
        'photo': open(photo_path, 'rb')
    }

    try:
        response = requests.post(API_ENDPOINT, data=payload, files=files)
        if response.status_code == 200:
            print(f"✔️ Emergência reportada com sucesso: {response.json()}")
        else:
            print(f"❌ Falha ao reportar: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"⚠️ Erro ao enviar emergência: {e}")

def process_video(video_path, window_name):
    global last_frame_blurred
    last_frame_blurred = None

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    frame_interval = 3
    frame_count = 0
    last_detection_time = None
    detection_delay = 5  # segundos

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        roi = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (250, 250))
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        confidence = prob[0][0]

        if DEBUG:
            print(f"[DEBUG] Predição: {pred}, Confiança: {round(confidence * 100, 2)}%")

        movimento_detectado = frame_has_motion(frame)

        if pred == "Accident" and confidence > 0.999 and movimento_detectado:
            now = datetime.now()
            if not last_detection_time or (now - last_detection_time).total_seconds() > detection_delay:
                last_detection_time = now

                photo_path = f"accident_{now.strftime('%Y%m%d_%H%M%S_%f')}.png"
                cv2.imwrite(photo_path, frame)
                print(f"[DEBUG] 🚨 Acidente detectado - salvando imagem: {photo_path}")

                send_emergency_report(window_name, photo_path)

                if os.name == 'nt':
                    winsound.Beep(1000, 500)

        # Mostrar na tela
        if confidence > 0.999:
            label = f"{pred} {round(confidence*100, 2)}%"
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, label, (20, 30), font, 1, (255, 255, 0), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyWindow(window_name)

def start_application():
    video_files = [
        'Camera1.mp4', 'Camera2.mp4', 'Car.mp4'
        # Adicione mais vídeos para teste
    ]

    threads = []

    for i, video_path in enumerate(video_files):
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"🔁 Iniciando processamento: {video_path}")
        thread = threading.Thread(target=process_video, args=(video_path, file_name))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_application()
