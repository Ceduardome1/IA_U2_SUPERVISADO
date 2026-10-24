import cv2
import numpy as np
from src.models.interfaces import IPreprocessor
from src.models.types import NDArrayFloat, NDArrayUInt8


class OpenCVDigitPreprocessor(IPreprocessor):
    """Pipeline de preprocesamiento de visión por computadora para normalizar
    el recorte de la cámara al formato esperado por la CNN: tensor (1, 1, 28, 28).
    """

    def __init__(self, min_active_pixels: int = 40) -> None:
        self._min_active_pixels = min_active_pixels
        self._last_processed_image: NDArrayUInt8 = np.zeros((28, 28), dtype=np.uint8)
        self._has_digit: bool = False

    @property
    def has_digit(self) -> bool:
        """Indica si el último recorte contenía suficientes píxeles de trazo."""
        return self._has_digit

    def get_last_processed_image(self) -> NDArrayUInt8:
        return self._last_processed_image

    def process_roi(self, roi: NDArrayUInt8) -> NDArrayFloat:
        """Convierte el ROI (BGR) de la cámara en un tensor normalizado (1, 1, 28, 28).
        
        Pasos del pipeline:
        1. Conversión a escala de grises.
        2. Filtro Gaussiano para reducir ruido.
        3. Umbralización de Otsu Invertida (Trazos oscuros de tinta -> Blanco, Papel -> Negro).
        4. Recorte de la caja delimitadora del trazo.
        5. Escalado proporcional manteniendo la relación de aspecto (caja máx 20x20).
        6. Centrado por momentos (Centro de Masa) en lienzo de 28x28.
        7. Normalización a [0.0, 1.0].
        """
        # 1. Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 2. Suavizado
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Binarización de Otsu Invertida
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        active_pixels = int(np.count_nonzero(thresh))
        if active_pixels < self._min_active_pixels:
            self._has_digit = False
            self._last_processed_image = np.zeros((28, 28), dtype=np.uint8)
            return np.zeros((1, 1, 28, 28), dtype=np.float32)

        self._has_digit = True

        # 4. Encontrar contorno del trazo
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        digit_crop = thresh[y : y + h, x : x + w]

        # 5. Redimensionar respetando relación de aspecto a máx 20x20
        if h > w:
            new_h = 20
            new_w = max(1, int(w * (20.0 / h)))
        else:
            new_w = 20
            new_h = max(1, int(h * (20.0 / w)))

        resized_digit = cv2.resize(
            digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

        # 6. Colocar en lienzo de 28x28 centrado
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_start = (28 - new_w) // 2
        y_start = (28 - new_h) // 2
        canvas[y_start : y_start + new_h, x_start : x_start + new_w] = resized_digit

        # Centrado fino por centro de masa
        centered_canvas = self._center_by_center_of_mass(canvas)
        self._last_processed_image = centered_canvas

        # 7. Normalización a [0.0, 1.0] y forma para CNN: (1, 1, 28, 28)
        normalized = (centered_canvas.astype(np.float32) / 255.0).reshape(1, 1, 28, 28)
        return normalized

    @staticmethod
    def _center_by_center_of_mass(image: NDArrayUInt8) -> NDArrayUInt8:
        """Ajusta la posición del dígito para alinear su centro de masa en (14, 14)."""
        moments = cv2.moments(image)
        if moments["m00"] == 0:
            return image

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        shift_x = int(np.round(14.0 - cx))
        shift_y = int(np.round(14.0 - cy))

        m_trans = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(
            image, m_trans, (28, 28), flags=cv2.INTER_NEAREST, borderValue=0
        )
        return shifted.astype(np.uint8)
