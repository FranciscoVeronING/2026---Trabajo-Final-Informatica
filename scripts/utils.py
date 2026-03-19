import numpy as np


def get_anchor_and_scale(pose_landmarks):
    """
    Calcula el ancla espacial (centro del pecho) y el factor de escala (distancia entre hombros).
    
    Esta normalización permite que las coordenadas extraídas sean invariantes 
    a la distancia de la persona respecto a la cámara y a su complexión física.

    Args:
        pose_landmarks (Any): Objeto de MediaPipe que contiene los hitos (landmarks) de la pose detectada.

    Returns:
        Tuple[np.ndarray, float]: 
            - anchor (np.array): Array 1D con las coordenadas (x, y, z) del punto central.
            - scale (float): Valor numérico de la distancia euclidiana entre los hombros.
    """
    if not pose_landmarks:
        return np.array([0.0, 0.0, 0.0]), 1.0
    
    l_shldr = pose_landmarks.landmark
    r_shldr = pose_landmarks.landmark
    
    #Normalización de Traslación: Punto medio entre los hombros
    anchor = np.array([(l_shldr.x + r_shldr.x) / 2, 
                       (l_shldr.y + r_shldr.y) / 2, 
                       (l_shldr.z + r_shldr.z) / 2])
    
    # Normalización de Escala: Distancia Euclidiana 2D entre hombros
    scale = np.sqrt((l_shldr.x - r_shldr.x)**2 + (l_shldr.y - r_shldr.y)**2)
    
    # Evitar división por cero si MediaPipe falla y pone los hombros en el mismo pixel
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