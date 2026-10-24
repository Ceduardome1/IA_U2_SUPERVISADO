from pathlib import Path
import time
from typing import Optional
from sklearn.model_selection import train_test_split
from src.config import AppConfig, DEFAULT_CONFIG
from src.models.augmenter import DataAugmenter
from src.models.classifier import DigitCNNClassifier
from src.models.dataset import MNISTDatasetLoader
from src.models.types import DatasetSplit, TrainingSummary
from src.views.cli_view import CLIView


class TrainController:
    """Controlador para el flujo de entrenamiento de la Red Convolucional (CNN)
    con monitoreo de Early Stopping didáctico y selección del punto óptimo.
    """

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._config = config
        self._loader = MNISTDatasetLoader(config)
        self._augmenter = DataAugmenter(config)
        self._cli_view = CLIView()

    def run(
        self,
        save_path: Optional[Path] = None,
        use_augmentation: bool = False,
        patience: Optional[int] = None,
    ) -> DigitCNNClassifier:
        """Ejecuta el ciclo con división Train/Val, Early Stopping y guardado del punto óptimo."""
        target_save = save_path or (
            self._config.augmented_model_path if use_augmentation else self._config.standard_model_path
        )
        pat = patience if patience is not None else self._config.early_stopping_patience

        mode_name = "CON DATA AUGMENTATION (Variaciones)" if use_augmentation else "ESTÁNDAR (Sin Augmentation)"
        self._cli_view.show_header(f"Entrenamiento CNN con Early Stopping [{mode_name}]")

        # Cargar datos locales
        base_split: DatasetSplit = self._loader.load_data(
            test_size=self._config.test_size,
            train_size=self._config.train_size,
        )

        if use_augmentation:
            self._cli_view.show_info("Aplicando Data Augmentation (+25% muestras con ruido y deformaciones)...")
            dataset_to_use: DatasetSplit = self._augmenter.augment_dataset(base_split, ratio_to_augment=0.25)
        else:
            dataset_to_use = base_split

        # Partición Train / Validation (85% Train / 15% Validation) para detectar sobreentrenamiento
        x_tr, x_val, y_tr, y_val = train_test_split(
            dataset_to_use.x_train,
            dataset_to_use.y_train,
            test_size=self._config.validation_ratio,
            random_state=self._config.random_state,
            stratify=dataset_to_use.y_train,
        )

        self._cli_view.show_info(f"División de datos para Early Stopping:")
        print(f"  • Muestras de Entrenamiento: {x_tr.shape[0]:,} (Ajuste de pesos mediante gradiente)")
        print(f"  • Muestras de Validación   : {x_val.shape[0]:,} (Identificación de punto óptimo / Overfitting)")
        print(f"  • Paciencia configurada    : {pat} épocas sin mejora permitidas")

        classifier = DigitCNNClassifier(self._config)

        start_time = time.time()
        summary: TrainingSummary = classifier.fit_with_early_stopping(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_val,
            y_val=y_val,
            patience=pat,
        )
        elapsed = time.time() - start_time

        # Mostrar resumen didáctico
        self._cli_view.show_early_stopping_summary(summary)
        self._cli_view.show_info(f"Tiempo total de entrenamiento: {elapsed:.2f} segundos.")

        # Persistir pesos del punto óptimo restaurado
        classifier.save(target_save)
        self._cli_view.show_info(f"Pesos de la época óptima #{summary.best_epoch} guardados en: {target_save}")

        return classifier
