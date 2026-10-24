from pathlib import Path
from typing import List
from src.config import AppConfig, DEFAULT_CONFIG
from src.views.cli_view import CLIView


class CleanController:
    """Controlador para limpiar los artefactos generados por el entrenamiento y aumento de datos."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._config = config
        self._cli_view = CLIView()

    def run(self, clean_augmented: bool = False) -> None:
        """Elimina los pesos entrenados y opcionalmente las imágenes aumentadas."""
        self._cli_view.show_header("Limpieza de Modelos y Artefactos de Entrenamiento")

        cleaned_files: List[Path] = []

        # 1. Limpiar modelos entrenados
        if self._config.models_dir.exists():
            for model_file in self._config.models_dir.glob("*.pt"):
                model_file.unlink()
                cleaned_files.append(model_file)
            for joblib_file in self._config.models_dir.glob("*.joblib"):
                joblib_file.unlink()
                cleaned_files.append(joblib_file)

        # 2. Limpiar imágenes aumentadas si se solicita
        if clean_augmented and self._config.augmented_dir.exists():
            import shutil
            shutil.rmtree(self._config.augmented_dir)
            self._cli_view.show_info(f"Carpeta de datos aumentados eliminada: {self._config.augmented_dir}")

        if cleaned_files:
            self._cli_view.show_info(f"Se eliminaron {len(cleaned_files)} archivo(s) de pesos de modelos:")
            for f in cleaned_files:
                print(f"  • {f.name}")
            self._cli_view.show_info("El entorno ha quedado limpio y listo para recrear el entrenamiento desde cero.")
        else:
            self._cli_view.show_info("No se encontraron modelos guardados para eliminar.")

        print(f"\n[NOTA] El dataset local '{self._config.local_dataset_file.name}' se mantiene intacto para no tener que descargarlo de nuevo.\n")
