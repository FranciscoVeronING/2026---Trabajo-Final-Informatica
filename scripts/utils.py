import pandas as pd
import numpy as np

def get_anchor_and_scale(pose_landmarks):
    """Calcula el ancla (centro del pecho) y la escala (distancia entre hombros)."""
    if not pose_landmarks:
        return np.array([0.0, 0.0, 0.0]), 1.0
    
    # ¡AQUÍ ESTABA EL ERROR! Faltaban los índices [1] y [2]
    l_shldr = pose_landmarks.landmark[1] # Hombro izquierdo
    r_shldr = pose_landmarks.landmark[2] # Hombro derecho
    
    # Normalización de Traslación: Punto medio entre los hombros
    anchor = np.array([(l_shldr.x + r_shldr.x) / 2, 
                       (l_shldr.y + r_shldr.y) / 2, 
                       (l_shldr.z + r_shldr.z) / 2])
    
    # Normalización de Escala: Distancia Euclidiana 2D entre hombros
    scale = np.sqrt((l_shldr.x - r_shldr.x)**2 + (l_shldr.y - r_shldr.y)**2)
    
    # Evitar división por cero si MediaPipe falla
    if scale < 1e-5: 
        scale = 1.0
        
    return anchor, scale


def normalize_landmarks(landmarks_flat, anchor, scale):
    """
    Normaliza un vector de coordenadas 1D restando el ancla y dividiendo por la escala.

    Args:
        landmarks_flat (np.array): Vector 1D plano que contiene las coordenadas espaciales.
        anchor (np.array): Vector 1D con el origen de coordenadas (x, y, z).
        scale (float): Factor de escala de referencia del torso.

    Returns:
        np.array: Vector 1D con los landmarks normalizados espacialmente.
    """
    # Si el vector es todo ceros (no se detectó la parte del cuerpo), lo dejamos así
    if np.all(landmarks_flat == 0):
        return landmarks_flat
    
    # Reconstruimos a (N, 3) para la resta matemática
    pts = landmarks_flat.reshape(-1, 3)
    pts_norm = (pts - anchor) / scale
    
    # Volvemos a aplanar a 1D
    return pts_norm.flatten()

def interpolar_secuencia(secuencia_data: np.ndarray) -> np.ndarray:
    """
    Rellena los fotogramas perdidos (donde MediaPipe devolvió ceros) 
    utilizando interpolación lineal temporal.
    
    Args:
        secuencia_data (np.ndarray): Matriz de landmarks original con posibles ceros.
        
    Returns:
        np.ndarray: Matriz con las trayectorias de movimiento suavizadas e imputadas.
    """
    # Convertimos la matriz a un DataFrame de Pandas para usar sus herramientas temporales
    df = pd.DataFrame(secuencia_data)
    
    # Reemplazamos los ceros por NaN (Not a Number) para que Pandas sepa dónde faltan datos
    df.replace(0.0, np.nan, inplace=True)
    
    # Aplicamos interpolación lineal. 'limit_direction='both'' copia el fotograma más cercano 
    # si la mano faltaba justo en el primer o último fotograma del video.
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    
    # Si una mano no apareció en TODO el video, quedarán NaNs. Los devolvemos a 0.0 por seguridad.
    df.fillna(0.0, inplace=True)
    
    return df.to_numpy()