import copy
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.config import AppConfig, DEFAULT_CONFIG
from src.models.interfaces import IClassifierModel
from src.models.types import (
    EpochLog,
    EvaluationMetrics,
    NDArrayFloat,
    NDArrayInt,
    PredictionResult,
    TrainingSummary,
)


class SimpleCNN(nn.Module):
    """Arquitectura Convolucional Didáctica (CNN).
    
    1. Conv1 (1 -> 16 filtros 3x3) + ReLU + MaxPool (28x28 -> 14x14)
    2. Conv2 (16 -> 32 filtros 3x3) + ReLU + MaxPool (14x14 -> 7x7)
    3. Capa Densa (32 * 7 * 7 = 1568 -> 64) + Dropout(0.25)
    4. Capa Salida (64 -> 10 dígitos)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EarlyStoppingTracker:
    """Rastreador didáctico del punto óptimo de entrenamiento (Early Stopping).
    
    Principios aplicados:
    - Subentrenamiento (Underfitting): La pérdida de validación sigue bajando; el modelo aún aprende.
    - Punto Óptimo (Sweet Spot): La pérdida de validación alcanza su mínimo histórico. Guardamos los pesos aquí.
    - Sobreentrenamiento (Overfitting): La pérdida de entrenamiento sigue bajando, pero la de
      validación empieza a subir o estancarse (memorización de ruido). Se activa la paciencia.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_weights: Optional[dict] = None
        self.should_stop = False

    def step(self, val_loss: float, epoch: int, model: nn.Module) -> Tuple[bool, int]:
        """Evalúa si la época actual es el nuevo punto óptimo o incrementa el contador de paciencia."""
        is_best = False
        # Si la pérdida de validación mejora al menos min_delta
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
            is_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return is_best, self.counter


class DigitCNNClassifier(IClassifierModel):
    """Clasificador Convolucional (CNN) con Early Stopping didáctico y selección del punto óptimo."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._config = config
        self._device = torch.device("cpu")
        self._network = SimpleCNN().to(self._device)
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit_with_early_stopping(
        self,
        x_train: NDArrayFloat,
        y_train: NDArrayInt,
        x_val: NDArrayFloat,
        y_val: NDArrayInt,
        patience: Optional[int] = None,
    ) -> TrainingSummary:
        """Entrena la CNN monitoreando el conjunto de validación para detectar el punto óptimo."""
        pat = patience if patience is not None else self._config.early_stopping_patience
        tracker = EarlyStoppingTracker(patience=pat, min_delta=self._config.early_stopping_min_delta)

        # Preparar DataLoaders
        train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
        train_loader = DataLoader(train_ds, batch_size=self._config.batch_size, shuffle=True)

        val_ds = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

        optimizer = torch.optim.Adam(self._network.parameters(), lr=self._config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        logs: List[EpochLog] = []

        print(f"\n{'Época':^7} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Paciencia':^11} | {'Estado / Diagnóstico':<25}")
        print("-" * 95)

        for epoch in range(1, self._config.epochs + 1):
            # 1. Fase de Entrenamiento
            self._network.train()
            train_loss_acc = 0.0
            train_correct = 0
            train_total = 0

            for bx, by in train_loader:
                bx, by = bx.to(self._device), by.to(self._device)
                optimizer.zero_grad()
                out = self._network(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

                train_loss_acc += loss.item() * bx.size(0)
                _, p = torch.max(out, 1)
                train_correct += (p == by).sum().item()
                train_total += bx.size(0)

            t_loss = train_loss_acc / train_total
            t_acc = (train_correct / train_total) * 100.0

            # 2. Fase de Validación
            self._network.eval()
            val_loss_acc = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self._device), by.to(self._device)
                    out = self._network(bx)
                    loss = criterion(out, by)
                    val_loss_acc += loss.item() * bx.size(0)
                    _, p = torch.max(out, 1)
                    val_correct += (p == by).sum().item()
                    val_total += bx.size(0)

            v_loss = val_loss_acc / val_total
            v_acc = (val_correct / val_total) * 100.0

            # 3. Evaluar Early Stopping
            is_best, current_patience = tracker.step(v_loss, epoch, self._network)

            # Diagnóstico didáctico para la consola
            if is_best:
                status_diag = "⭐ NUEVO PUNTO ÓPTIMO"
                patience_str = f"0/{pat} (Reset)"
            else:
                patience_str = f"{current_patience}/{pat}"
                if current_patience >= pat:
                    status_diag = "⛔ DETENIDO (Sobreentrenando)"
                else:
                    status_diag = "⚠️ Sin mejora (Alerta Overfit)"

            print(f"{epoch:^7d} | {t_loss:^10.4f} | {t_acc:9.2f}% | {v_loss:^10.4f} | {v_acc:9.2f}% | {patience_str:^11} | {status_diag:<25}")

            logs.append(
                EpochLog(
                    epoch=epoch,
                    train_loss=t_loss,
                    train_acc=t_acc,
                    val_loss=v_loss,
                    val_acc=v_acc,
                    is_best=is_best,
                    patience_counter=current_patience,
                )
            )

            if tracker.should_stop:
                print("-" * 95)
                print(f"[EARLY STOPPING] Criterio cumplido: Se agotó la paciencia ({pat} épocas sin reducción de Val Loss).")
                break

        print("-" * 95)
        # 4. Restaurar los pesos del mejor punto óptimo identificado
        if tracker.best_weights is not None:
            self._network.load_state_dict(tracker.best_weights)
            print(f"[RESTAURACIÓN] Pesos del modelo restaurados a la ÉPOCA ÓPTIMA #{tracker.best_epoch} (Val Loss: {tracker.best_loss:.4f}).")

        self._is_fitted = True

        return TrainingSummary(
            best_epoch=tracker.best_epoch,
            best_val_loss=tracker.best_loss,
            stopped_epoch=logs[-1].epoch,
            stopped_early=tracker.should_stop,
            total_epochs=len(logs),
            logs=logs,
        )

    def fit(self, x_train: NDArrayFloat, y_train: NDArrayInt) -> None:
        """Entrenamiento simple estándar por compatibilidad."""
        split_idx = int(len(x_train) * 0.85)
        xt, xv = x_train[:split_idx], x_train[split_idx:]
        yt, yv = y_train[:split_idx], y_train[split_idx:]
        self.fit_with_early_stopping(xt, yt, xv, yv)

    def predict(self, x: NDArrayFloat) -> PredictionResult:
        if not self._is_fitted:
            raise RuntimeError("El modelo CNN no ha sido entrenado ni cargado.")

        self._network.eval()
        if x.shape == (1, 784) or x.ndim == 1:
            x_input = x.reshape(1, 1, 28, 28)
        elif x.ndim == 3 and x.shape == (1, 28, 28):
            x_input = x.reshape(1, 1, 28, 28)
        else:
            x_input = x

        tensor_x = torch.from_numpy(x_input).float().to(self._device)

        with torch.no_grad():
            logits = self._network(tensor_x)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float32)

        predicted_digit: int = int(np.argmax(probabilities))
        confidence: float = float(probabilities[predicted_digit])

        is_valid = confidence >= self._config.min_confidence_threshold
        status = "OK" if is_valid else "Confianza baja"

        return PredictionResult(
            digit=predicted_digit,
            confidence=confidence,
            probabilities=probabilities,
            is_valid=is_valid,
            status_message=status,
        )

    def evaluate(self, x_test: NDArrayFloat, y_test: NDArrayInt) -> EvaluationMetrics:
        if not self._is_fitted:
            raise RuntimeError("El modelo CNN no ha sido entrenado ni cargado.")

        self._network.eval()
        tensor_x = torch.from_numpy(x_test).float()
        tensor_y = torch.from_numpy(y_test).long()
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(self._device)
                out = self._network(bx)
                _, preds = torch.max(out, 1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(by.numpy().tolist())

        y_pred = np.array(all_preds, dtype=np.int64)
        y_true = np.array(all_targets, dtype=np.int64)

        accuracy = float(np.mean(y_pred == y_true))
        cm: NDArrayInt = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(10)), zero_division=0
        )

        return EvaluationMetrics(
            accuracy=accuracy,
            confusion_matrix=cm,
            per_class_precision=precision.astype(np.float32),
            per_class_recall=recall.astype(np.float32),
            total_samples=len(y_test),
        )

    def save(self, file_path: Path) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._network.state_dict(), file_path)
        self._is_fitted = True

    def load(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"No existe el archivo de pesos en {file_path}")
        self._network.load_state_dict(torch.load(file_path, map_location=self._device, weights_only=True))
        self._network.eval()
        self._is_fitted = True
