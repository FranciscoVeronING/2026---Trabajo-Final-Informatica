import os


import cv2
import numpy as np
import mediapipe as mp
import glob
from tqdm import tqdm
from utils import get_anchor_and_scale, interpolar_secuencia, normalize_landmarks

def process_video_landmarks(video_path: str, holistic: any) -> np.ndarray:
    """Procesa un video y extrae landmarks normalizados de pose y manos."""
    cap = cv2.VideoCapture(video_path)
    sequence_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        anchor, scale = get_anchor_and_scale(results.pose_landmarks)

        # Omitimos extraer la cara para reducir a 225 features
        pose_raw = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
        lh_raw = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh_raw = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

        pose_norm = normalize_landmarks(pose_raw, anchor, scale)
        lh_norm = normalize_landmarks(lh_raw, anchor, scale)
        rh_norm = normalize_landmarks(rh_raw, anchor, scale)

        frame_data = np.concatenate([pose_norm, lh_norm, rh_norm])
        sequence_data.append(frame_data)

    cap.release()
    return interpolar_secuencia(np.array(sequence_data))    

def extraer_dataset_completo(dir_entrada: str, dir_salida: str) -> None:
    """Recorre las carpetas, procesa los videos etiquetados y los guarda como.npy"""
    os.makedirs(dir_salida, exist_ok=True)
    clases_señas = [d for d in os.listdir(dir_entrada) if os.path.isdir(os.path.join(dir_entrada, d))]
    
    mp_holistic = mp.solutions.holistic
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for clase in clases_señas:
            print(f"\nProcesando clase: {clase}")
            ruta_clase_entrada = os.path.join(dir_entrada, clase)
            ruta_clase_salida = os.path.join(dir_salida, clase)
            os.makedirs(ruta_clase_salida, exist_ok=True)
            
            videos = glob.glob(os.path.join(ruta_clase_entrada, "*.mp4"))
            
            for video_path in tqdm(videos, desc=f"Video"):
                nombre_archivo = os.path.basename(video_path).replace('.mp4', '.npy')
                ruta_guardado = os.path.join(ruta_clase_salida, nombre_archivo)
                
                if os.path.exists(ruta_guardado):
                    continue 
                
                try:
                    landmarks = process_video_landmarks(video_path, holistic)
                    if len(landmarks) > 0:
                        np.save(ruta_guardado, landmarks)
                except Exception as e:
                    print(f"Error procesando {nombre_archivo}: {e}")

if __name__ == "__main__":
    DIR_ENTRADA = r"C:\Users\franc\Documents\GitHub\2026---Trabajo-Final-Informatica\entrenamiento\dataset"
    DIR_SALIDA = r"C:\Users\franc\Documents\GitHub\2026---Trabajo-Final-Informatica\entrenamiento\dataset_procesado_npy"
    extraer_dataset_completo(DIR_ENTRADA, DIR_SALIDA)