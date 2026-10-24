from src.models.types import DatasetSplit, EvaluationMetrics, PredictionResult, TrainingSummary


class CLIView:
    """Vista de consola para reportes didácticos y métricas de evaluación."""

    @staticmethod
    def show_header(title: str) -> None:
        print("\n" + "=" * 70)
        print(f"  {title.upper()}")
        print("=" * 70)

    @staticmethod
    def show_info(message: str) -> None:
        print(f"[INFO] {message}")

    @staticmethod
    def show_dataset_info(split: DatasetSplit) -> None:
        print("\n--- Partición de Datos (MNIST para CNN) ---")
        print(f"• Conjunto de Entrenamiento: {split.x_train.shape[0]:,} imágenes (Tensor: {split.x_train.shape[1:]})")
        print(f"• Conjunto de Pruebas      : {split.x_test.shape[0]:,} imágenes (Tensor: {split.x_test.shape[1:]})")
        print(f"• Rango de valores         : [{split.x_train.min():.1f}, {split.x_train.max():.1f}] (Normalizado)")
        print(f"• Distribución de clases   : 10 dígitos (0 al 9) estratificados")

    @staticmethod
    def show_early_stopping_summary(summary: TrainingSummary) -> None:
        print("\n" + "=" * 70)
        print("  DIAGNÓSTICO DIDÁCTICO DEL ENTRENAMIENTO (EARLY STOPPING)")
        print("=" * 70)
        if summary.stopped_early:
            print(f"• Estado: DETENCIÓN TEMPRANA ACTIVADA (Evitó sobreentrenamiento / overfitting).")
            print(f"• Época de parada   : #{summary.stopped_epoch}")
        else:
            print(f"• Estado: Completó todas las {summary.total_epochs} épocas programadas.")

        print(f"• PUNTO ÓPTIMO      : Época #{summary.best_epoch}")
        print(f"• Pérdida de Valid. : {summary.best_val_loss:.4f} (Mínimo histórico alcanzado)")
        print("\nInterpretación didáctica:")
        print("  1. Antes de la época óptima -> Subentrenamiento (el modelo aún aprendía patrones).")
        print("  2. En la época óptima       -> Máxima generalización (pesos guardados).")
        print("  3. Después de la época óptima -> Comienza la memorización de ruido particular del set.")
        print("=" * 70)

    @staticmethod
    def show_evaluation_report(metrics: EvaluationMetrics) -> None:
        print("\n" + "-" * 70)
        print(f"  RESULTADOS DE EVALUACIÓN CNN EN CONJUNTO DE TEST ({metrics.total_samples} MUESTRAS)")
        print("-" * 70)
        print(f"Accuracy Global: {metrics.accuracy * 100:.2f}%\n")

        print(f"{'Dígito':^8} | {'Precisión':^12} | {'Recall':^12}")
        print("-" * 40)
        for digit in range(10):
            prec = metrics.per_class_precision[digit] * 100
            rec = metrics.per_class_recall[digit] * 100
            print(f"{digit:^8} | {prec:10.2f}% | {rec:10.2f}%")
        print("-" * 40)

        print("\nMatriz de Confusión (Filas: Real, Columnas: Predicho):")
        header = "   " + " ".join(f"{d:>5}" for d in range(10))
        print(header)
        print("   " + "-" * (6 * 10))
        for i, row in enumerate(metrics.confusion_matrix):
            row_str = " ".join(f"{val:>5}" for val in row)
            print(f"{i:>2}| {row_str}")

    @staticmethod
    def show_prediction(result: PredictionResult) -> None:
        print(f"\n[INFERENCIA CNN] Dígito: {result.digit} | Confianza: {result.confidence * 100:.2f}% | Estado: {result.status_message}")
        print("Distribución de probabilidades:")
        for d, prob in enumerate(result.probabilities):
            bar = "█" * int(prob * 30)
            print(f"  [{d}]: {prob * 100:5.1f}% | {bar}")

    @staticmethod
    def show_theoretical_justification() -> None:
        print("\n" + "=" * 70)
        print("  JUSTIFICACIÓN DIDÁCTICA: CNN, DATA AUGMENTATION Y EARLY STOPPING")
        print("=" * 70)
        print("""
1. ¿Por qué una Red Neuronal Convolucional (CNN)?
   • Preserva la estructura 2D de la imagen.
   • Los filtros convolucionales (kernels 3x3) aprenden patrones visuales locales:
     - Capa 1: Bordes, esquinas y trazos rectos/curvos.
     - Capa 2: Uniones de trazos, bucles y aperturas numéricas.
   • El submuestreo (MaxPooling 2x2) aporta invarianza a pequeñas traslaciones.

2. Identificación del Punto Óptimo (Early Stopping):
   • Subentrenamiento (Underfitting): La red no ha iterado lo suficiente y su error es alto.
   • Sobreentrenamiento (Overfitting): La red memoriza los píxeles particulares de entrenamiento
     y pierde capacidad de generalizar con datos nuevos.
   • Early Stopping divide un 15% de validación. Cuando la pérdida de validación deja de mejorar
     durante N épocas consecutivas (paciencia configurable), detiene el ciclo y restaura
     automáticamente los mejores pesos históricos.

3. Principio de Aumento de Datos (Data Augmentation):
   • Genera variaciones morfológicas (ruido, deformaciones, rotaciones) para que la red
     aprenda a reconocer trazos sin importar la calidad de la cámara o grosor del lápiz.
""")
        print("=" * 70)
