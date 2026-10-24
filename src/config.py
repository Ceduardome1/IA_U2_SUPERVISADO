from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Configuración inmutable y tipada para todo el proyecto."""

    # Rutas del sistema
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    local_dataset_file: Path = data_dir / "mnist_local.npz"
    augmented_dir: Path = data_dir / "augmented_digits"
    models_dir: Path = base_dir / "models_saved"
    standard_model_path: Path = models_dir / "mnist_cnn_standard.pt"
    augmented_model_path: Path = models_dir / "mnist_cnn_augmented.pt"
    default_model_path: Path = standard_model_path

    # Parámetros del Dataset
    train_size: int = 60000
    test_size: int = 10000
    validation_ratio: float = 0.15  # 15% del conjunto de entrenamiento para validación (Early Stopping)
    random_state: int = 42

    # Parámetros de la Red Neuronal Convolucional (CNN)
    epochs: int = 25  # Número máximo de épocas permitidas
    batch_size: int = 64
    learning_rate: float = 0.001

    # Parámetros didácticos de Early Stopping (Detención Temprana)
    early_stopping_patience: int = 3  # Número de épocas sin mejora antes de detener
    early_stopping_min_delta: float = 0.001  # Mínima mejora requerida en la pérdida de validación

    # Parámetros de Cámara y Preprocesamiento
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    roi_size: int = 240  # Tamaño del cuadrado de foco en píxeles
    auto_sample_interval_frames: int = 15  # Cada 15 fotogramas en modo auto
    min_confidence_threshold: float = 0.50  # Confianza mínima para reportar dígito


DEFAULT_CONFIG = AppConfig()
