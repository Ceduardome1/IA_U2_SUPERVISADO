from pathlib import Path
from typing import Optional
from src.config import AppConfig, DEFAULT_CONFIG
from src.models.classifier import DigitCNNClassifier
from src.models.dataset import MNISTDatasetLoader
from src.models.types import DatasetSplit, EvaluationMetrics
from src.views.cli_view import CLIView


class EvalController:
    """Controlador para evaluar y comparar rigurosamente modelos CNN."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._config = config
        self._loader = MNISTDatasetLoader(config)
        self._cli_view = CLIView()

    def run(
        self,
        model_path: Optional[Path] = None,
        classifier: Optional[DigitCNNClassifier] = None,
    ) -> Optional[EvaluationMetrics]:
        """Carga la CNN y evalúa su rendimiento en las 10,000 imágenes de prueba."""
        target_path = model_path or self._config.standard_model_path

        # Validación estricta: No permitir evaluar sin haber entrenado
        if classifier is None:
            if not target_path.exists():
                self._cli_view.show_header("Error: Modelo No Encontrado")
                self._cli_view.show_info(
                    f"No existe el archivo de modelo en '{target_path}'.\n"
                    "  [VALIDACIÓN REQUERIDA]: Debes entrenar la red antes de evaluar.\n"
                    "  Ejecuta primero: './run.sh src/main.py train' o './run.sh src/main.py train --augmented'\n"
                )
                return None

            self._cli_view.show_info(f"Cargando pesos de la CNN desde {target_path}...")
            classifier = DigitCNNClassifier(self._config)
            classifier.load(target_path)

        self._cli_view.show_info("Cargando partición de prueba local...")
        split: DatasetSplit = self._loader.load_data(
            test_size=self._config.test_size,
            train_size=self._config.train_size,
        )

        self._cli_view.show_info("Calculando métricas y matriz de confusión en la CNN...")
        metrics: EvaluationMetrics = classifier.evaluate(split.x_test, split.y_test)

        self._cli_view.show_evaluation_report(metrics)
        return metrics

    def compare(self) -> None:
        """Evalúa y compara ambos modelos (Estándar vs Aumentado) lado a lado."""
        std_path = self._config.standard_model_path
        aug_path = self._config.augmented_model_path

        self._cli_view.show_header("Comparativa Didáctica: Estándar vs Data Augmentation")

        # Validación estricta para la comparativa
        missing_models = []
        if not std_path.exists():
            missing_models.append(f"• Modelo Estándar ('{std_path.name}') -> Ejecuta: './run.sh src/main.py train'")
        if not aug_path.exists():
            missing_models.append(f"• Modelo Aumentado ('{aug_path.name}') -> Ejecuta: './run.sh src/main.py train --augmented'")

        if missing_models:
            self._cli_view.show_info("No se puede realizar la comparación porque faltan modelos por entrenar:")
            for m in missing_models:
                print(f"  {m}")
            print()
            return

        split: DatasetSplit = self._loader.load_data(
            test_size=self._config.test_size,
            train_size=self._config.train_size,
        )

        print("\n1. Evaluando Modelo Estándar (Sin Augmentation)...")
        std_model = DigitCNNClassifier(self._config)
        std_model.load(std_path)
        m_std = std_model.evaluate(split.x_test, split.y_test)

        print("\n2. Evaluando Modelo Aumentado (Con Variaciones y Ruido)...")
        aug_model = DigitCNNClassifier(self._config)
        aug_model.load(aug_path)
        m_aug = aug_model.evaluate(split.x_test, split.y_test)

        print("\n" + "=" * 65)
        print("         TABLA COMPARATIVA EN CONJUNTO DE PRUEBAS")
        print("=" * 65)
        print(f"{'Métrica':<25} | {'Estándar':^15} | {'Con Augmentation':^15}")
        print("-" * 65)
        print(f"{'Accuracy Global':<25} | {m_std.accuracy * 100:14.2f}% | {m_aug.accuracy * 100:14.2f}%")
        print("-" * 65)
        print("Precisión por Dígito:")
        for d in range(10):
            p_std = m_std.per_class_precision[d] * 100
            p_aug = m_aug.per_class_precision[d] * 100
            print(f"  Dígito [{d}]:                | {p_std:14.2f}% | {p_aug:14.2f}%")
        print("=" * 65 + "\n")
