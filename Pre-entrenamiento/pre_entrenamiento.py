import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# ==========================================
# UTILIDADES Y DATASET
# ==========================================
class EarlyStopping:
    """Detiene el entrenamiento si la pérdida de validación no mejora."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, path: str) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), path)
            print(f"   [*] Mejor modelo base guardado (MSE: {val_loss:.4f})")
        else:
            self.counter += 1
            print(f"   [!] Sin mejoras. Contador: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class UnlabeledSkeletonDataset(Dataset):
    """Carga tensores .npy para el Autoencoder."""
    def __init__(self, data_dir: str, max_frames: int = 100):
        self.data_dir = data_dir
        self.archivos = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.archivos)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ruta_completa = os.path.join(self.data_dir, self.archivos[idx])
        secuencia = np.load(ruta_completa)
        frames_actuales, features = secuencia.shape
        
        if frames_actuales < self.max_frames:
            padding = np.zeros((self.max_frames - frames_actuales, features))
            secuencia = np.vstack((secuencia, padding))
        else:
            secuencia = secuencia[:self.max_frames, :]
            
        return torch.tensor(secuencia, dtype=torch.float32)

# ==========================================
# ARQUITECTURA (Sincronizada con Fine-Tuning)
# ==========================================
class PositionalEncoding(nn.Module):
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
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, num_classes: int = 1, dropout_rate: float = 0.5):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Arquitectura sincronizada con el aumento de regularización
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4, 
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim) 
        self.classification_head = nn.Linear(hidden_dim, num_classes) 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))

    def forward(self, x: torch.Tensor, mode: str = 'pretrain', mask_indices: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        if mode == 'pretrain' and mask_indices is not None:
            mask_expanded = self.mask_token.expand(batch_size, seq_len, -1)
            x = torch.where(mask_indices, mask_expanded, x)

        x = x.permute(1, 0, 2) 
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        if mode == 'pretrain':
            return self.reconstruction_head(x)
        return self.classification_head(x.mean(dim=1))

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Hiperparámetros reducidos para evitar Overfitting
    INPUT_DIM = 225  
    HIDDEN_DIM = 128
    EPOCHS_PRETRAIN = 100
    MAX_FRAMES = 100 
    BATCH_SIZE = 32
    MASK_RATIO = 0.15 
    PATIENCE = 15
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo: {device}")
    
    ruta_npy = r"C:\Users\franc\Documents\GitHub\2026---Trabajo-Final-Informatica\Pre-entrenamiento\dataset_procesado_npy"
    dataset_completo = UnlabeledSkeletonDataset(ruta_npy, max_frames=MAX_FRAMES)
    
    train_size = int(0.8 * len(dataset_completo))
    val_size = len(dataset_completo) - train_size
    train_dataset, val_dataset = random_split(dataset_completo, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Instanciamos el modelo con la arquitectura reducida y mayor Dropout
    model = SkeletonTransformer(
        input_dim=INPUT_DIM, 
        hidden_dim=HIDDEN_DIM, 
        num_heads=4, 
        num_layers=2,
        dropout_rate=0.5
    ).to(device)
    
    # Weight decay elevado mantenido
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # MSELoss normal (Sin label_smoothing)
    loss_fn = nn.MSELoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=PATIENCE)
    ruta_mejor_modelo = "transformer_cinematica_preentrenado.pth"
    
    print("\n--- INICIANDO PRE-ENTRENAMIENTO (AUTOENCODER) ---")
    train_history, val_history = [], []
    
    for epoch in range(EPOCHS_PRETRAIN):
        model.train()
        train_loss = 0.0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            b_size, seq_len, _ = batch_data.shape
            mask = torch.rand(b_size, seq_len, 1, device=device) < MASK_RATIO
            
            reconstructed = model(batch_data, mode='pretrain', mask_indices=mask)
            loss = loss_fn(reconstructed[mask.squeeze(-1)], batch_data[mask.squeeze(-1)])
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_history.append(avg_train_loss)
        
        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                b_size, seq_len, _ = batch_data.shape
                mask = torch.rand(b_size, seq_len, 1, device=device) < MASK_RATIO
                
                reconstructed = model(batch_data, mode='pretrain', mask_indices=mask)
                loss = loss_fn(reconstructed[mask.squeeze(-1)], batch_data[mask.squeeze(-1)])
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_history.append(avg_val_loss)
        
        print(f"Época {epoch+1}/{EPOCHS_PRETRAIN} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model, ruta_mejor_modelo)
        
        if early_stopping.early_stop:
            print("\n[!] Detención Temprana activada.")
            break
            
    # Graficar
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Train MSE')
    plt.plot(val_history, label='Validation MSE')
    plt.title("Curva de Aprendizaje - Pre-entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.savefig("evaluacion_preentrenamiento_7.png")
    plt.close()
    print("[*] ¡Pre-entrenamiento finalizado óptimamente!")