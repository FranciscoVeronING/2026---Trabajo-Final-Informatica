import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# UTILIDADES Y DATASET
# ==========================================
class EarlyStopping:
    """Detiene el entrenamiento si la pérdida de validación no mejora para evitar Overfitting."""
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, path: str):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), path)
            print(f"   [*] Nuevo mejor modelo guardado (Loss: {val_loss:.4f})")
        else:
            self.counter += 1
            print(f"   [!] Sin mejoras. Contador EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class LabeledSkeletonDataset(Dataset):
    """Carga arrays.npy para Fine-Tuning."""
    def __init__(self, data_dir: str, max_frames: int = 100):
        self.max_frames = max_frames
        self.clases = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {clase: idx for idx, clase in enumerate(self.clases)}
        
        self.archivos = []
        self.etiquetas = []
        
        for clase in self.clases:
            rutas_npy = glob.glob(os.path.join(data_dir, clase, "*.npy"))
            for ruta in rutas_npy:
                self.archivos.append(ruta)
                self.etiquetas.append(self.class_to_idx[clase])

    def __len__(self) -> int:
        return len(self.archivos)

    def __getitem__(self, idx: int):
        secuencia = np.load(self.archivos[idx])
        frames_actuales, features = secuencia.shape
        
        if frames_actuales < self.max_frames:
            padding = np.zeros((self.max_frames - frames_actuales, features))
            secuencia = np.vstack((secuencia, padding))
        else:
            secuencia = secuencia[:self.max_frames, :]
            
        return torch.tensor(secuencia, dtype=torch.float32), torch.tensor(self.etiquetas[idx], dtype=torch.long)
    

def augment_batch_3d(batch_data: torch.Tensor, noise_std: float = 0.01, scale_range: tuple = (0.9, 1.1)) -> torch.Tensor:
    """
    Aplica Data Augmentation espacial en tiempo real a un lote de landmarks 3D.
    
    Args:
        batch_data (torch.Tensor): Tensor de forma (Batch, Frames, 225).
        noise_std (float): Desviación estándar para el ruido Gaussiano.
        scale_range (tuple): Rango (min, max) para el factor de escala.
        
    Returns:
        torch.Tensor: Tensor aumentado con la misma forma original.
    """
    b_size, seq_len, features = batch_data.shape
    device = batch_data.device
    
    # Redimensionamos temporalmente a (Batch, Frames, 75 puntos, 3 ejes) para operar en 3D
    x_3d = batch_data.view(b_size, seq_len, -1, 3)
    
    # 1. Escalado Aleatorio (un factor de escala distinto por cada secuencia del batch)
    scales = torch.empty(b_size, 1, 1, 1, device=device).uniform_(*scale_range)
    x_augmented = x_3d * scales
    
    # 2. Ruido Gaussiano (Jittering) aplicado a cada coordenada individualmente
    noise = torch.randn_like(x_augmented, device=device) * noise_std
    x_augmented = x_augmented + noise
    
    # Devolvemos a la forma plana original (Batch, Frames, 225)
    return x_augmented.view(b_size, seq_len, features)

# ==========================================
# ARQUITECTURA
# ==========================================
class PositionalEncoding(nn.Module):
    """Inyecta información sobre el orden secuencial de los fotogramas."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class SkeletonTransformer(nn.Module):
    """
    Modelo basado en Transformer Encoder para la clasificación de secuencias de lenguaje de señas.
    
    Args:
        input_dim (int): Dimensión de las características de entrada (225 landmarks).
        hidden_dim (int): Dimensión oculta para las proyecciones y el Transformer.
        num_heads (int): Número de cabezales de atención.
        num_layers (int): Número de capas del Transformer Encoder.
        num_classes (int): Cantidad de clases a predecir (100 señas).
        dropout_rate (float): Probabilidad de enmascaramiento para la regularización.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier_dropout = nn.Dropout(p=dropout_rate)
        self.classification_head = nn.Linear(hidden_dim, num_classes) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante del modelo.
        
        Args:
            x (torch.Tensor): Tensor de entrada con forma (Batch, Secuencia, Features).
            
        Returns:
            torch.Tensor: Logits de clasificación con forma (Batch, num_classes).
        """
        x = x.permute(1, 0, 2) 
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        
        # Agrupamiento temporal (Mean Pooling)
        x_pooled = x.mean(dim=1)
        
        # Regularización final antes de emitir los logits
        x_dropped = self.classifier_dropout(x_pooled)
        
        return self.classification_head(x_dropped)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Hiperparámetros actualizados
    INPUT_DIM = 225  
    HIDDEN_DIM = 128
    EPOCHS = 200
    MAX_FRAMES = 100 
    BATCH_SIZE = 16
    PATIENCE = 15 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo: {device}")
    
    DIR_DATOS = r"C:\Users\franc\Documents\GitHub\2026---Trabajo-Final-Informatica\entrenamiento\dataset_procesado_npy"
    dataset_completo = LabeledSkeletonDataset(DIR_DATOS, max_frames=MAX_FRAMES)
    NUM_CLASSES = len(dataset_completo.clases)
    
    train_size = int(0.8 * len(dataset_completo))
    test_size = len(dataset_completo) - train_size
    train_dataset, test_dataset = random_split(dataset_completo, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SkeletonTransformer(INPUT_DIM, HIDDEN_DIM, num_heads=4, num_layers=2, num_classes=NUM_CLASSES).to(device)
    
    ruta_pesos_pre = r"C:\Users\franc\Documents\GitHub\2026---Trabajo-Final-Informatica\scripts\transformer_cinematica_preentrenado.pth"
    if os.path.exists(ruta_pesos_pre):
        print("[*] Cargando conocimiento base (Transfer Learning)...")
        pretrained_dict = torch.load(ruta_pesos_pre, map_location=device, weights_only=True)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict) 
    else:
        print("[!] No se encontró pre-entrenamiento. Entrenando desde cero.")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=PATIENCE)
    ruta_mejor_modelo = "transformer_clasificador_señas_best.pth"
    
    print("\n--- INICIANDO FINE-TUNING ---")
    train_loss_history, val_loss_history = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correctos_train, muestras_train = 0.0, 0, 0
        
        # Barra de progreso en vivo
        bucle_lotes = tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS}", leave=False)
        
        for batch_data, batch_labels in bucle_lotes:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            batch_data = augment_batch_3d(batch_data, noise_std=0.015, scale_range=(0.85, 1.15))

            optimizer.zero_grad()
            
            logits = model(batch_data)
            loss = loss_fn(logits, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicciones = torch.argmax(logits, dim=1)
            correctos_train += (predicciones == batch_labels).sum().item()
            muestras_train += batch_labels.size(0)
            
        avg_train_loss = train_loss / len(train_loader)
        acc_train = (correctos_train / muestras_train) * 100
        train_loss_history.append(avg_train_loss)
        
        model.eval()
        val_loss, correctos_val, muestras_val = 0.0, 0, 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                logits = model(batch_data)
                loss = loss_fn(logits, batch_labels)
                
                val_loss += loss.item()
                predicciones = torch.argmax(logits, dim=1)
                correctos_val += (predicciones == batch_labels).sum().item()
                muestras_val += batch_labels.size(0)
                
        avg_val_loss = val_loss / len(test_loader)
        acc_val = (correctos_val / muestras_val) * 100
        val_loss_history.append(avg_val_loss)
        
        print(f"Época {epoch+1} | Train Loss: {avg_train_loss:.4f} (Acc: {acc_train:.2f}%) | Val Loss: {avg_val_loss:.4f} (Acc: {acc_val:.2f}%)")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model, ruta_mejor_modelo)
        
        if early_stopping.early_stop:
            print("\n[!] Detención Temprana activada.")
            break
            
    print("\n--- EVALUANDO EL MEJOR MODELO ---")
    model.load_state_dict(torch.load(ruta_mejor_modelo, weights_only=True))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            preds = torch.argmax(model(batch_data), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title("Curva de Aprendizaje")
    plt.legend()
    plt.grid(True)
    plt.savefig("curva_finetuning_v7.png")
    plt.close()
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset_completo.clases)
    fig, ax = plt.subplots(figsize=(15, 12))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=90)
    plt.tight_layout()
    plt.savefig("matriz_confusion_final_v7.png")
    plt.close()
    print("[*] Fin del proceso.")