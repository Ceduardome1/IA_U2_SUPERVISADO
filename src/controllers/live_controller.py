from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from src.config import AppConfig, DEFAULT_CONFIG
from src.models.classifier import DigitCNNClassifier
from src.models.preprocessor import OpenCVDigitPreprocessor
from src.models.types import PredictionResult
from src.views.camera_view import CameraView
from src.views.cli_view import CLIView


class LiveController:
    """Controlador que orquesta la captura de video en vivo, preprocesamiento e inferencia con CNN."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._config = config
        self._preprocessor = OpenCVDigitPreprocessor()
        self._camera_view = CameraView(config)
        self._cli_view = CLIView()

    def run(
        self,
        camera_id: Optional[int] = None,
        model_path: Optional[Path] = None,
        classifier: Optional[DigitCNNClassifier] = None,
    ) -> None:
        """Bucle principal de ejecución con cámara web."""
        target_path = model_path or self._config.default_model_path
        cam_id = camera_id if camera_id is not None else self._config.camera_id

        # Validación estricta: Bloquear inicio de cámara si el modelo no está entrenado
        if classifier is None:
            if not target_path.exists():
                self._cli_view.show_header("Error: No se permite escanear sin entrenar")
                self._cli_view.show_info(
                    f"No se encontró ningún modelo entrenado en '{target_path}'.\n"
                    "  [VALIDACIÓN REQUERIDA]: La cámara web no puede realizar inferencias sin un modelo entrenado.\n"
                    "  Por favor entrena el modelo ejecutando cualquiera de estos comandos:\n"
                    "    • Modelo estándar : './run.sh src/main.py train'\n"
                    "    • Modelo aumentado: './run.sh src/main.py train --augmented'\n"
                )
                return

            self._cli_view.show_info(f"Cargando modelo entrenado desde: {target_path}")
            classifier = DigitCNNClassifier(self._config)
            classifier.load(target_path)

        # Iniciar cámara
        self._cli_view.show_info(f"Conectando a la cámara web (índice: {cam_id})...")
        cap = cv2.VideoCapture(cam_id)

        if not cap.isOpened():
            self._cli_view.show_info(
                f"No se pudo abrir la cámara web con índice {cam_id}. "
                "Verifica que el dispositivo esté conectado y accesible o especifica otro índice con '--camera-id <N>'."
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera_height)

        self._camera_view.open_window()
        self._cli_view.show_header("Modo Camara Web Activo (CNN)")
        self._cli_view.show_info("Haz clic en los botones en pantalla o usa los atajos:")
        self._cli_view.show_info(" • Botón '[ESPACIO] Capturar' para clasificar el fotograma actual.")
        self._cli_view.show_info(" • Botón '[A] Modo Auto' para alternar muestreo continuo.")
        self._cli_view.show_info(" • Botón '[Q] Salir' para cerrar la aplicación.")

        is_auto_mode = False
        frame_counter = 0
        last_prediction: Optional[PredictionResult] = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    self._cli_view.show_info("Error leyendo fotograma de la cámara.")
                    break

                # 1. Obtener coordenadas del área de foco (ROI)
                roi_coords = self._camera_view.get_roi_coordinates(frame.shape)
                x1, y1, x2, y2 = roi_coords
                roi = frame[y1:y2, x1:x2]

                # 2. Preprocesar ROI a tensor convolucional (1, 1, 28, 28)
                tensor_28x28 = self._preprocessor.process_roi(roi)
                pip_img = self._preprocessor.get_last_processed_image()
                has_digit = self._preprocessor.has_digit

                # 3. Decidir si se ejecuta inferencia por intervalo en modo auto
                trigger_prediction = False

                if is_auto_mode:
                    frame_counter += 1
                    if frame_counter >= self._config.auto_sample_interval_frames:
                        frame_counter = 0
                        trigger_prediction = True

                # Renderizar fotograma y procesar eventos
                key = self._camera_view.render_frame(
                    raw_frame=frame,
                    roi_coords=roi_coords,
                    last_prediction=last_prediction,
                    pip_image=pip_img,
                    is_auto_mode=is_auto_mode,
                    has_digit=has_digit,
                )

                # Procesar clics sobre los botones en pantalla
                btn_action = self._camera_view.pop_clicked_action()
                if btn_action == "capture":
                    trigger_prediction = True
                elif btn_action == "toggle_auto":
                    is_auto_mode = not is_auto_mode
                    frame_counter = 0
                    self._cli_view.show_info(f"Modo auto-muestreo: {'ACTIVADO' if is_auto_mode else 'DESACTIVADO'}")
                elif btn_action == "quit":
                    break

                # Procesar teclas del teclado
                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord(" "), ord("c"), ord("C")):
                    trigger_prediction = True
                elif key in (ord("a"), ord("A")):
                    is_auto_mode = not is_auto_mode
                    frame_counter = 0
                    self._cli_view.show_info(f"Modo auto-muestreo: {'ACTIVADO' if is_auto_mode else 'DESACTIVADO'}")

                # Realizar inferencia con CNN
                if trigger_prediction:
                    if has_digit:
                        last_prediction = classifier.predict(tensor_28x28)
                        self._cli_view.show_prediction(last_prediction)
                    else:
                        last_prediction = PredictionResult(
                            digit=-1,
                            confidence=0.0,
                            probabilities=np.zeros(10, dtype=np.float32),
                            is_valid=False,
                            status_message="No se detecta trazo en el foco",
                        )

        finally:
            cap.release()
            self._camera_view.close_window()
            self._cli_view.show_info("Cámara web cerrada correctamente.")
