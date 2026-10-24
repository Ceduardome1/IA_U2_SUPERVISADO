from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import numpy.typing as npt

# Tipos estrictos para tensores y matrices
NDArrayFloat = npt.NDArray[np.float32]
NDArrayUInt8 = npt.NDArray[np.uint8]
NDArrayInt = npt.NDArray[np.int64]


@dataclass(frozen=True)
class DatasetSplit:
    """Contenedor inmutable para las particiones de entrenamiento y prueba."""
    x_train: NDArrayFloat
    y_train: NDArrayInt
    x_test: NDArrayFloat
    y_test: NDArrayInt

    def summary(self) -> str:
        return (
            f"Train: {self.x_train.shape[0]} muestras, Dim: {self.x_train.shape[1:]} | "
            f"Test: {self.x_test.shape[0]} muestras, Dim: {self.x_test.shape[1:]}"
        )


@dataclass(frozen=True)
class PredictionResult:
    """Resultado tipado de una inferencia."""
    digit: int
    confidence: float
    probabilities: NDArrayFloat
    is_valid: bool = True
    status_message: str = "OK"


@dataclass(frozen=True)
class EvaluationMetrics:
    """Métricas didácticas de evaluación del clasificador."""
    accuracy: float
    confusion_matrix: NDArrayInt
    per_class_precision: NDArrayFloat
    per_class_recall: NDArrayFloat
    total_samples: int


@dataclass(frozen=True)
class EpochLog:
    """Registro por época para monitoreo didáctico de Early Stopping."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    is_best: bool
    patience_counter: int


@dataclass(frozen=True)
class TrainingSummary:
    """Resumen final del proceso de entrenamiento con Early Stopping."""
    best_epoch: int
    best_val_loss: float
    stopped_epoch: int
    stopped_early: bool
    total_epochs: int
    logs: List[EpochLog]
