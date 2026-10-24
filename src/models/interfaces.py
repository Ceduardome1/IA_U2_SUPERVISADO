from pathlib import Path
from typing import Protocol, runtime_checkable
from src.models.types import (
    DatasetSplit,
    EvaluationMetrics,
    NDArrayFloat,
    NDArrayInt,
    NDArrayUInt8,
    PredictionResult,
)


@runtime_checkable
class IClassifierModel(Protocol):
    """Protocolo formal para el clasificador convolucional de dígitos."""

    def fit(self, x_train: NDArrayFloat, y_train: NDArrayInt) -> None:
        """Entrena la CNN con los datos suministrados."""
        ...

    def predict(self, x: NDArrayFloat) -> PredictionResult:
        """Predice la clase y probabilidades para una muestra de entrada (1, 1, 28, 28)."""
        ...

    def evaluate(self, x_test: NDArrayFloat, y_test: NDArrayInt) -> EvaluationMetrics:
        """Calcula métricas detalladas en el conjunto de prueba."""
        ...

    def save(self, file_path: Path) -> None:
        """Serializa los pesos del modelo a disco."""
        ...

    def load(self, file_path: Path) -> None:
        """Carga los pesos serializados desde disco."""
        ...

    @property
    def is_fitted(self) -> bool:
        """Indica si el modelo ya fue entrenado o cargado."""
        ...


@runtime_checkable
class IPreprocessor(Protocol):
    """Protocolo para el preprocesamiento de imágenes de cámara hacia el formato MNIST."""

    def process_roi(self, roi: NDArrayUInt8) -> NDArrayFloat:
        """Transforma un recorte de cámara (BGR) en un tensor normalizado de (1, 1, 28, 28)."""
        ...

    def get_last_processed_image(self) -> NDArrayUInt8:
        """Retorna la última imagen 28x28 procesada (para visualización en pantalla)."""
        ...
