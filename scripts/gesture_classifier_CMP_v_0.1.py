import cv2
import numpy as np
import mediapipe as mp
from utils import get_anchor_and_scale, normalize_landmarks

mp_holistic = mp.solutions.holistic

def process_video_landmarks(video_path, holistic):
    """
    Procesa un video y extrae exclusivamente los landmarks normalizados para cada fotograma.
    
    Su uso sera el crear los inputs para un Transformer puro, ya que descarta los píxeles y 
    se enfoca únicamente en la cinemática topológica.

    Args:
        video_path (str): Ruta absoluta o relativa al archivo de video.
        holistic (Any): Instancia inicializada del modelo MediaPipe Holistic.

    Returns:
        np.ndarray: Matriz de forma (Fotogramas, N_Landmarks) con la secuencia temporal de coordenadas.
    """
    cap = cv2.VideoCapture(video_path)
    sequence_data = ()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Obtener la referencia de normalización basada en el torso
        anchor, scale = get_anchor_and_scale(results.pose_landmarks)

        # Extraer puntos crudos (rellenando con ceros si no hay)
        pose_raw = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(24*3)
        lh_raw = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh_raw = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        face_raw = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

        # Aplicar normalización espacial a cada conjunto
        pose_norm = normalize_landmarks(pose_raw, anchor, scale)
        lh_norm = normalize_landmarks(lh_raw, anchor, scale)
        rh_norm = normalize_landmarks(rh_raw, anchor, scale)
        face_norm = normalize_landmarks(face_raw, anchor, scale)

        # Unir todo en un solo vector normalizado
        frame_data = np.concatenate([pose_norm, lh_norm, rh_norm, face_norm])
        sequence_data.append(frame_data)

    cap.release()
    return np.array(sequence_data)
