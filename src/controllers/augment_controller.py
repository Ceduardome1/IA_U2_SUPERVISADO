from pathlib import Path
from typing import Optional
from src.config import AppConfig, DEFAULT_CONFIG
from src.models.augmenter import DataAugmenter
from src.models.dataset import MNISTDatasetLoader
from src.models.types import DatasetSplit
from src.views.cli_view import CLIView


class AugmentController:
    """Controlador para la fase independiente de Aumento de Datos (Data Augmentation)."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._config = config
        self._loader = MNISTDatasetLoader(config)
        self._augmenter = DataAugmenter(config)
        self._cli_view = CLIView()

    def run(self, samples_per_digit: int = 5, target_dir: Optional[Path] = None) -> Path:
        """Exporta muestras didácticas organizadas por carpetas de clase con todas sus variaciones."""
        self._cli_view.show_header("Fase Separada: Generación y Agrupación de Datos Aumentados")
        self._cli_view.show_info(
            "Cargando dataset local y aplicando transformaciones: ruido, rotaciones, deformaciones y volteado..."
        )

        split: DatasetSplit = self._loader.load_data(
            test_size=self._config.test_size,
            train_size=self._config.train_size,
        )

        dest_dir = self._augmenter.export_augmented_folder(
            x_samples=split.x_test,
            y_labels=split.y_test,
            samples_per_digit=samples_per_digit,
            target_dir=target_dir,
        )

        self._cli_view.show_info(f"Variaciones generadas y guardadas en: {dest_dir}")
        self._cli_view.show_info("Estructura: data/augmented_digits/digito_{0..9}/sample_X_{variacion}.png")
        return dest_dir
