from typing import Callable, Dict, Optional, Tuple
import cv2
import numpy as np
from src.config import AppConfig, DEFAULT_CONFIG
from src.models.types import NDArrayUInt8, PredictionResult


class Button:
    """Representa un botón interactivo dibujado en pantalla con coordenadas y callback."""

    def __init__(
        self,
        name: str,
        label: str,
        x: int,
        y: int,
        w: int,
        h: int,
        color_normal: Tuple[int, int, int] = (60, 60, 60),
        color_active: Tuple[int, int, int] = (0, 180, 80),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.name = name
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color_normal = color_normal
        self.color_active = color_active
        self.text_color = text_color
        self.is_active: bool = False

    def contains(self, px: int, py: int) -> bool:
        """Determina si un punto (clic del mouse) está dentro del botón."""
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, frame: NDArrayUInt8) -> None:
        """Renderiza el botón en el fotograma."""
        bg = self.color_active if self.is_active else self.color_normal
        # Fondo del botón con bordes redondeados simulados
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), bg, -1)
        border_color = (0, 255, 180) if self.is_active else (120, 120, 120)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), border_color, 1)

        # Centrar texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        text_size = cv2.getTextSize(self.label, font, font_scale, thickness)[0]
        tx = self.x + (self.w - text_size[0]) // 2
        ty = self.y + (self.h + text_size[1]) // 2

        cv2.putText(frame, self.label, (tx, ty), font, font_scale, self.text_color, thickness)


class CameraView:
    """Vista de OpenCV con área de foco, PiP e interfaz gráfica interactiva con botones."""

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._config = config
        self._window_name = "Clasificador MNIST CNN - Camara Web"
        self._last_clicked_action: Optional[str] = None
        self._buttons: Dict[str, Button] = {}
        self._init_buttons()

    def _init_buttons(self) -> None:
        """Inicializa la botonera inferior interactiva."""
        y_pos = self._config.camera_height - 38
        h_btn = 30

        # Botón 1: Capturar
        self._buttons["capture"] = Button(
            name="capture",
            label="[ESPACIO] Capturar",
            x=15,
            y=y_pos,
            w=145,
            h=h_btn,
            color_normal=(70, 70, 70),
            color_active=(0, 160, 230),
        )

        # Botón 2: Alternar Modo Auto/Manual
        self._buttons["toggle_auto"] = Button(
            name="toggle_auto",
            label="[A] Modo Auto: OFF",
            x=170,
            y=y_pos,
            w=150,
            h=h_btn,
            color_normal=(70, 70, 70),
            color_active=(0, 180, 80),
        )

        # Botón 3: Salir
        self._buttons["quit"] = Button(
            name="quit",
            label="[Q] Salir",
            x=330,
            y=y_pos,
            w=95,
            h=h_btn,
            color_normal=(50, 50, 140),
            color_active=(30, 30, 200),
        )

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Escucha clics del ratón sobre los botones interactivos."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn_name, btn in self._buttons.items():
                if btn.contains(x, y):
                    self._last_clicked_action = btn_name
                    break

    def open_window(self) -> None:
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

    def close_window(self) -> None:
        cv2.destroyAllWindows()

    def get_roi_coordinates(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Calcula las coordenadas (x1, y1, x2, y2) del cuadrado de foco central."""
        height, width, _ = frame_shape
        size = min(self._config.roi_size, min(height, width) - 60)
        cx, cy = width // 2, (height - 30) // 2

        x1 = cx - size // 2
        y1 = cy - size // 2
        x2 = x1 + size
        y2 = y1 + size
        return x1, y1, x2, y2

    def pop_clicked_action(self) -> Optional[str]:
        """Obtiene y limpia la última acción disparada por clic en botón."""
        action = self._last_clicked_action
        self._last_clicked_action = None
        return action

    def render_frame(
        self,
        raw_frame: NDArrayUInt8,
        roi_coords: Tuple[int, int, int, int],
        last_prediction: Optional[PredictionResult],
        pip_image: Optional[NDArrayUInt8],
        is_auto_mode: bool,
        has_digit: bool,
    ) -> int:
        """Dibuja todos los elementos en el fotograma y procesa la tecla pulsada."""
        display = raw_frame.copy()
        x1, y1, x2, y2 = roi_coords

        # 1. Dibujar el área de foco (ROI)
        roi_color = (0, 255, 0) if has_digit else (0, 200, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), roi_color, 2)

        # Esquinas decorativas
        corner_len = 20
        thick = 3
        cv2.line(display, (x1, y1), (x1 + corner_len, y1), (0, 255, 0), thick)
        cv2.line(display, (x1, y1), (x1, y1 + corner_len), (0, 255, 0), thick)
        cv2.line(display, (x2, y1), (x2 - corner_len, y1), (0, 255, 0), thick)
        cv2.line(display, (x2, y1), (x2, y1 + corner_len), (0, 255, 0), thick)
        cv2.line(display, (x1, y2), (x1 + corner_len, y2), (0, 255, 0), thick)
        cv2.line(display, (x1, y2), (x1, y2 - corner_len), (0, 255, 0), thick)
        cv2.line(display, (x2, y2), (x2 - corner_len, y2), (0, 255, 0), thick)
        cv2.line(display, (x2, y2), (x2 - corner_len, y2), (0, 255, 0), thick)

        cv2.putText(
            display,
            "AREA DE FOCO (Coloca aqui el digito)",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

        # 2. Panel superior con resultado de la predicción
        cv2.rectangle(display, (0, 0), (display.shape[1], 50), (25, 25, 25), -1)

        if last_prediction is not None and last_prediction.is_valid:
            pred_text = f"CNN DIGITO: {last_prediction.digit} | Confianza: {last_prediction.confidence * 100:.1f}%"
            cv2.putText(
                display,
                pred_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 100),
                2,
            )
        elif last_prediction is not None and not last_prediction.is_valid:
            cv2.putText(
                display,
                f"DIGITO: {last_prediction.digit} (?) Confianza baja: {last_prediction.confidence * 100:.1f}%",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 165, 255),
                2,
            )
        else:
            cv2.putText(
                display,
                "LISTO: Clic en 'Capturar' o presiona ESPACIO",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (220, 220, 220),
                2,
            )

        # 3. Barra inferior interactiva con Botones
        h = display.shape[0]
        cv2.rectangle(display, (0, h - 45), (display.shape[1], h), (20, 20, 20), -1)

        # Actualizar estado de botón de modo auto
        self._buttons["toggle_auto"].is_active = is_auto_mode
        self._buttons["toggle_auto"].label = "[A] Modo Auto: ON" if is_auto_mode else "[A] Modo Auto: OFF"

        for btn in self._buttons.values():
            btn.draw(display)

        # 4. Inset Picture-in-Picture (PiP) con la imagen 28x28 procesada
        if pip_image is not None and pip_image.shape == (28, 28):
            pip_size = 90
            pip_resized = cv2.resize(pip_image, (pip_size, pip_size), interpolation=cv2.INTER_NEAREST)
            pip_bgr = cv2.cvtColor(pip_resized, cv2.COLOR_GRAY2BGR)

            px_end = display.shape[1] - 15
            px_start = px_end - pip_size
            py_start = 65
            py_end = py_start + pip_size

            cv2.rectangle(display, (px_start - 2, py_start - 20), (px_end + 2, py_end + 2), (50, 50, 50), -1)
            cv2.putText(
                display,
                "Entrada CNN 28x28",
                (px_start - 5, py_start - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
            )
            display[py_start:py_end, px_start:px_end] = pip_bgr
            cv2.rectangle(display, (px_start, py_start), (px_end, py_end), (0, 255, 255), 1)

        cv2.imshow(self._window_name, display)
        key = cv2.waitKey(1) & 0xFF
        return key
