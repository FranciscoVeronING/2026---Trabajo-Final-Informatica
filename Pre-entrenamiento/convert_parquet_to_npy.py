import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# Le indicamos a Python que agregue la carpeta principal (padre) a su ruta de búsqueda
ruta_principal = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ruta_principal)

from scripts.utils import interpolar_secuencia

def parquet_a_secuencia(ruta_parquet: str) -> np.ndarray:
    """
    Lee un archivo Parquet y lo convierte en una matriz NumPy.
    Extrae exclusivamente la pose y las manos para evitar el sobreajuste.
    
    Args:
        ruta_parquet (str): Ruta al archivo de datos crudos.
        
    Returns:
        np.ndarray: Matriz de forma (Fotogramas, 225).
    """
    df = pd.read_parquet(ruta_parquet)
    df.fillna(0.0, inplace=True) # Rellenar nulos con 0.0 para evitar errores matemáticos
    
    frames = df['frame'].unique()
    secuencia_normalizada = []
    
    for frame in frames:
        df_frame = df[df['frame'] == frame]
        
        # Extraer solo pose y manos (33 + 21 + 21 = 75 landmarks * 3 = 225 coordenadas)
        pose_df = df_frame[df_frame['type'] == 'pose'][['x', 'y', 'z']].values.flatten()
        lh_df = df_frame[df_frame['type'] == 'left_hand'][['x', 'y', 'z']].values.flatten()
        rh_df = df_frame[df_frame['type'] == 'right_hand'][['x', 'y', 'z']].values.flatten()
        
        # Unir en un solo vector de 225 dimensiones
        frame_data = np.concatenate([pose_df, lh_df, rh_df])
        secuencia_normalizada.append(frame_data)
        
    return interpolar_secuencia(np.array(secuencia_normalizada))

def procesar_dataset_parquet(directorio_entrada: str, directorio_salida: str) -> None:
    """Recorre las carpetas, convierte los parquets y los guarda como.npy"""
    os.makedirs(directorio_salida, exist_ok=True)
    rutas_parquets = glob.glob(os.path.join(directorio_entrada, "*", "*.parquet"))
    
    print(f"Se encontraron {len(rutas_parquets)} archivos. Iniciando conversión...")
    
    for ruta in tqdm(rutas_parquets, desc="Convirtiendo Parquets"):
        nombre_archivo = os.path.basename(ruta).replace('.parquet', '.npy')
        ruta_guardado = os.path.join(directorio_salida, nombre_archivo)
        
        if os.path.exists(ruta_guardado):
            continue
            
        try:
            matriz_numpy = parquet_a_secuencia(ruta)
            np.save(ruta_guardado, matriz_numpy)
        except Exception as e:
            print(f"Error procesando {ruta}: {e}")

if __name__ == "__main__":
    DIR_ENTRADA = r"C:\Users\franc\Documents\GitHub\2026---Trabajo-Final-Informatica\Pre-entrenamiento\train_landmark_files"
    DIR_SALIDA = r"C:\Users\franc\Documents\GitHub\2026---Trabajo-Final-Informatica\Pre-entrenamiento\dataset_procesado_npy"
    procesar_dataset_parquet(DIR_ENTRADA, DIR_SALIDA)