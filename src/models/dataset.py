from pathlib import Path
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import AppConfig, DEFAULT_CONFIG
from src.models.types import DatasetSplit, NDArrayFloat, NDArrayInt


class MNISTDatasetLoader:
    """Cargador 100% local y particionador didáctico del dataset MNIST.
    
    Lee directamente el archivo empaquetado 'data/mnist_local.npz' sin depender
    de descargas de red ni servicios externos.
    """

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._config = config
        self._local_file = config.local_dataset_file

    def load_data(self, test_size: int = 10000, train_size: int = 60000) -> DatasetSplit:
        """Carga el dataset desde el archivo local y retorna la partición Train/Test."""
        if not self._local_file.exists():
            raise FileNotFoundError(
                f"No se encontró el dataset local en '{self._local_file}'. "
                "Verifica que el archivo exista en la carpeta data/."
            )

        archive = np.load(self._local_file)
        x_raw: NDArrayFloat = archive["x"].astype(np.float32)
        y_raw: NDArrayInt = archive["y"].astype(np.int64)

        # Reshape a tensor espacial 4D (N, 1, 28, 28) requerido por la CNN
        x_images = x_raw.reshape(-1, 1, 28, 28)

        total_requested = train_size + test_size
        if total_requested < len(x_images):
            x_images = x_images[:total_requested]
            y_raw = y_raw[:total_requested]

        # Partición estratificada para mantener balance exacto entre los dígitos 0 al 9
        x_tr, x_te, y_tr, y_te = train_test_split(
            x_images,
            y_raw,
            test_size=test_size,
            train_size=train_size,
            random_state=self._config.random_state,
            stratify=y_raw,
        )

        return DatasetSplit(
            x_train=np.ascontiguousarray(x_tr, dtype=np.float32),
            y_train=np.ascontiguousarray(y_tr, dtype=np.int64),
            x_test=np.ascontiguousarray(x_te, dtype=np.float32),
            y_test=np.ascontiguousarray(y_te, dtype=np.int64),
        )
