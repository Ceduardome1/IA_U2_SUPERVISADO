from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from src.config import AppConfig, DEFAULT_CONFIG
from src.models.types import DatasetSplit, NDArrayFloat, NDArrayInt, NDArrayUInt8


class DataAugmenter:
    """Generador didáctico de variaciones morfológicas y transformaciones de datos
    (Aumento de datos / Data Augmentation) para fortalecer la robustez del modelo.
    
    Transformaciones didácticas:
    1. original: Dígito canónico.
    2. ruido: Ruido gaussiano para simular sensores ruidosos o poca iluminación.
    3. rotado_izq_15 / rotado_der_15: Inclinación caligráfica (-15° y +15°).
    4. deformado_grueso / deformado_fino: Dilatación/erosión para variedad de trazo.
    5. deformado_corte: Deformación afín por corte lateral (cursiva).
    6. volteado_horizontal: Espejado horizontal.
    """

    def __init__(self, config: AppConfig = DEFAULT_CONFIG) -> None:
        self._config = config

    def generate_variations(self, image_28x28: NDArrayUInt8) -> Dict[str, NDArrayUInt8]:
        """Aplica las transformaciones a una imagen 28x28 en uint8 [0, 255]."""
        variations: Dict[str, NDArrayUInt8] = {}

        # 1. Original
        variations["original"] = image_28x28.copy()

        # 2. Ruido (Simulación de ruido de sensor)
        noise = np.random.normal(0, 35, image_28x28.shape).astype(np.int16)
        noisy = np.clip(image_28x28.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        variations["ruido"] = noisy

        # 3. Rotación izquierda (-15°)
        center = (14, 14)
        rot_mat_l = cv2.getRotationMatrix2D(center, -15, 1.0)
        variations["rotado_izq_15"] = cv2.warpAffine(image_28x28, rot_mat_l, (28, 28), borderValue=0)

        # 4. Rotación derecha (+15°)
        rot_mat_r = cv2.getRotationMatrix2D(center, 15, 1.0)
        variations["rotado_der_15"] = cv2.warpAffine(image_28x28, rot_mat_r, (28, 28), borderValue=0)

        # 5. Deformado grueso (Dilatación morfológica)
        kernel = np.ones((2, 2), np.uint8)
        variations["deformado_grueso"] = cv2.dilate(image_28x28, kernel, iterations=1)

        # 6. Deformado fino (Erosión morfológica)
        variations["deformado_fino"] = cv2.erode(image_28x28, kernel, iterations=1)

        # 7. Deformación por corte afín (Shearing cursivo)
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
        pts2 = np.float32([[5 + 3, 5], [20 + 3, 5], [5 - 3, 20]])
        shear_mat = cv2.getAffineTransform(pts1, pts2)
        variations["deformado_corte"] = cv2.warpAffine(image_28x28, shear_mat, (28, 28), borderValue=0)

        # 8. Volteado horizontal (Flip)
        variations["volteado_horizontal"] = cv2.flip(image_28x28, 1)

        return variations

    def export_augmented_folder(
        self,
        x_samples: NDArrayFloat,
        y_labels: np.ndarray,
        samples_per_digit: int = 5,
        target_dir: Optional[Path] = None,
    ) -> Path:
        """Agrupa en carpetas por dígito (0 al 9) los dígitos con sus variaciones generadas."""
        export_path = target_dir or self._config.augmented_dir
        export_path.mkdir(parents=True, exist_ok=True)

        counts: Dict[int, int] = {d: 0 for d in range(10)}

        for img_tensor, label_int in zip(x_samples, y_labels):
            digit = int(label_int)
            if counts[digit] >= samples_per_digit:
                continue

            digit_dir = export_path / f"digito_{digit}"
            digit_dir.mkdir(parents=True, exist_ok=True)

            sample_idx = counts[digit]
            counts[digit] += 1

            if img_tensor.ndim == 3:  # (1, 28, 28)
                img_28 = (img_tensor[0] * 255.0).astype(np.uint8)
            else:
                img_28 = (img_tensor.reshape(28, 28) * 255.0).astype(np.uint8)

            variations = self.generate_variations(img_28)

            for var_name, var_img in variations.items():
                file_name = f"sample_{sample_idx}_{var_name}.png"
                cv2.imwrite(str(digit_dir / file_name), var_img)

            if all(c >= samples_per_digit for c in counts.values()):
                break

        return export_path

    def augment_dataset(
        self,
        base_split: DatasetSplit,
        ratio_to_augment: float = 0.25,
    ) -> DatasetSplit:
        """Genera un nuevo DatasetSplit agregando variaciones morfológicas al conjunto de entrenamiento.
        
        Permite entrenar y comparar de forma separada:
        - Modelo Estándar (sin data augmentation)
        - Modelo Aumentado (con data augmentation)
        """
        n_augment = int(len(base_split.x_train) * ratio_to_augment)
        indices = np.random.choice(len(base_split.x_train), size=n_augment, replace=False)

        new_x_list: List[NDArrayFloat] = []
        new_y_list: List[int] = []

        # Lista de transformaciones seleccionadas para robustez
        transforms = [
            "ruido",
            "rotado_izq_15",
            "rotado_der_15",
            "deformado_grueso",
            "deformado_fino",
            "deformado_corte",
        ]

        for idx in indices:
            img_tensor = base_split.x_train[idx]  # (1, 28, 28)
            label = int(base_split.y_train[idx])

            img_uint8 = (img_tensor[0] * 255.0).astype(np.uint8)
            variations = self.generate_variations(img_uint8)

            # Tomar una transformación aleatoria
            chosen_var = np.random.choice(transforms)
            var_img = variations[chosen_var]

            var_float = (var_img.astype(np.float32) / 255.0).reshape(1, 28, 28)
            new_x_list.append(var_float)
            new_y_list.append(label)

        aug_x = np.stack(new_x_list, axis=0).astype(np.float32)
        aug_y = np.array(new_y_list, dtype=np.int64)

        combined_x = np.concatenate([base_split.x_train, aug_x], axis=0)
        combined_y = np.concatenate([base_split.y_train, aug_y], axis=0)

        # Barajar el conjunto aumentado
        shuffle_idx = np.random.permutation(len(combined_x))
        final_x_train = np.ascontiguousarray(combined_x[shuffle_idx], dtype=np.float32)
        final_y_train = np.ascontiguousarray(combined_y[shuffle_idx], dtype=np.int64)

        return DatasetSplit(
            x_train=final_x_train,
            y_train=final_y_train,
            x_test=base_split.x_test,
            y_test=base_split.y_test,
        )
