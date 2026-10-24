import argparse
from pathlib import Path
import sys

# Agregar la raíz del proyecto al sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import AppConfig, DEFAULT_CONFIG
from src.controllers.augment_controller import AugmentController
from src.controllers.clean_controller import CleanController
from src.controllers.eval_controller import EvalController
from src.controllers.live_controller import LiveController
from src.controllers.train_controller import TrainController
from src.views.cli_view import CLIView


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sistema Didáctico de Detección de Dígitos MNIST con CNN y Early Stopping (Patrón MVC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # Comando 'train' (Estándar o con Augmentation y Paciencia configurable)
    p_train = subparsers.add_parser("train", help="Entrena la CNN con Early Stopping y detección de punto óptimo")
    p_train.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs, help="Máximo número de épocas")
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size, help="Tamaño de lote")
    p_train.add_argument("--patience", type=int, default=DEFAULT_CONFIG.early_stopping_patience, help="Paciencia (épocas sin mejora)")
    p_train.add_argument("--augmented", action="store_true", help="Entrena usando Data Augmentation (variaciones)")

    # Comando 'eval'
    p_eval = subparsers.add_parser("eval", help="Evalúa el modelo CNN en las 10,000 imágenes de prueba")
    p_eval.add_argument("--model-path", type=str, default=None, help="Ruta de pesos a evaluar")

    # Comando 'compare'
    subparsers.add_parser("compare", help="Compara lado a lado el modelo Estándar vs el modelo con Data Augmentation")

    # Comando 'augment'
    p_aug = subparsers.add_parser("augment", help="Fase separada: exporta imágenes con variaciones agrupadas en carpetas")
    p_aug.add_argument("--samples-per-digit", type=int, default=5, help="Muestras base por cada dígito (0..9)")

    # Comando 'live'
    p_live = subparsers.add_parser("live", help="Inicia la cámara web con botones en pantalla para interactuar")
    p_live.add_argument("--camera-id", type=int, default=DEFAULT_CONFIG.camera_id, help="Índice de la cámara")
    p_live.add_argument("--use-augmented", action="store_true", help="Usa el modelo entrenado con Data Augmentation")

    # Comando 'clean'
    p_clean = subparsers.add_parser("clean", help="Limpia los modelos entrenados para poder recrear el entrenamiento")
    p_clean.add_argument("--augmented", action="store_true", help="Elimina también la carpeta de datos aumentados")

    # Comando 'theory'
    subparsers.add_parser("theory", help="Muestra la justificación didáctica (CNN, Data Augmentation, Early Stopping)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = DEFAULT_CONFIG

    if args.command == "train":
        if hasattr(args, "epochs") or hasattr(args, "batch_size") or hasattr(args, "patience"):
            config = AppConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                early_stopping_patience=args.patience,
            )
        controller = TrainController(config)
        controller.run(use_augmentation=args.augmented, patience=args.patience)

    elif args.command == "eval":
        model_p = Path(args.model_path) if args.model_path else config.standard_model_path
        controller = EvalController(config)
        controller.run(model_path=model_p)

    elif args.command == "compare":
        controller = EvalController(config)
        controller.compare()

    elif args.command == "augment":
        controller = AugmentController(config)
        controller.run(samples_per_digit=args.samples_per_digit)

    elif args.command == "clean":
        controller = CleanController(config)
        controller.run(clean_augmented=args.augmented)

    elif args.command == "live":
        controller = LiveController(config)
        chosen_model = config.augmented_model_path if args.use_augmented else config.standard_model_path
        controller.run(camera_id=args.camera_id, model_path=chosen_model)

    elif args.command == "theory":
        CLIView.show_theoretical_justification()


if __name__ == "__main__":
    main()
