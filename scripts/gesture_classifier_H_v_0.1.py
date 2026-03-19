import cv2
import numpy as np
import mediapipe as mp
from utils import get_anchor_and_scale, normalize_landmarks

mp_holistic = mp.solutions.holistic

def get_bounding_box(landmarks, image_shape, padding=20):
    """
    Calcula el cuadro delimitador (bounding box) en coordenadas de píxeles a partir de los landmarks.

    Args:
        landmarks (Any): Hitos (landmarks) detectados por MediaPipe para una región específica (ej. mano).
        image_shape (Tuple[int, int, int]): Dimensiones del fotograma original (Alto, Ancho, Canales).
        padding (int, optional): Píxeles extra para dar margen al recorte. Por defecto es 20.

    Returns:
        Tuple[int, int, int, int]: Coordenadas (xmin, ymin, xmax, ymax) del cuadro delimitador.
    """
    h, w, _ = image_shape
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    
    xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
    ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)
    
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(w, xmax + padding)
    ymax = min(h, ymax + padding)
    
    return xmin, ymin, xmax, ymax


def process_video_hybrid(video_path, holistic, crop_size=(64, 64)):
    """
    Procesa un video extrayendo landmarks normalizados y recortes (crops) en escala de grises de las manos.
    
    Args:
        video_path (str): Ruta absoluta o relativa al archivo de video.
        holistic (Any): Instancia inicializada del modelo MediaPipe Holistic.
        crop_size (Tuple[int, int], optional): Resolución final de los recortes de las manos. Por defecto es (64, 64).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - sequence_landmarks: Matriz temporal con los landmarks normalizados.
            - sequence_images: Matriz temporal con los tensores de imágenes recortadas.
    """
    cap = cv2.VideoCapture(video_path)
    sequence_landmarks = ()
    sequence_images = ()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        results = holistic.process(image_rgb)
        
        # Referencia de normalización para los vectores
        anchor, scale = get_anchor_and_scale(results.pose_landmarks)
        
        # Extracción y normalización de Landmarks
        lh_raw = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh_raw = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        lh_norm = normalize_landmarks(lh_raw, anchor, scale)
        rh_norm = normalize_landmarks(rh_raw, anchor, scale)
        
        frame_landmarks = np.concatenate([lh_norm, rh_norm])
        sequence_landmarks.append(frame_landmarks)
        
        # Extracción de Recortes en Escala de Grises (usando coords crudas de results)
        lh_crop = np.zeros(crop_size, dtype=np.uint8)
        rh_crop = np.zeros(crop_size, dtype=np.uint8)
        
        if results.left_hand_landmarks:
            xmin, ymin, xmax, ymax = get_bounding_box(results.left_hand_landmarks, frame.shape)
            if xmax > xmin and ymax > ymin:
                crop = image_gray[ymin:ymax, xmin:xmax]
                lh_crop = cv2.resize(crop, crop_size)
                
        if results.right_hand_landmarks:
            xmin, ymin, xmax, ymax = get_bounding_box(results.right_hand_landmarks, frame.shape)
            if xmax > xmin and ymax > ymin:
                crop = image_gray[ymin:ymax, xmin:xmax]
                rh_crop = cv2.resize(crop, crop_size)
        
        frame_crops = np.stack([lh_crop, rh_crop], axis=-1) 
        sequence_images.append(frame_crops)

    cap.release()
    return np.array(sequence_landmarks), np.array(sequence_images)
