# Clasificador y Detector de Dígitos MNIST con CNN y Cámara Web (Patrón MVC)

Sistema didáctico de visión por computadora y aprendizaje supervisado para clasificación y detección de dígitos manuscritos (0 al 9) mediante cámara web, implementado bajo el patrón **Modelo-Vista-Controlador (MVC)** con **tipado estricto (`typing`)**.

---

## 1. Características Principales

1. **Dataset 100% Local:**
   - Ubicado en [`data/mnist_local.npz`](file:///mnt/datos/Code/Pyton/IA/U2/SUPERVISADO/data/mnist_local.npz) (70,000 muestras completas).
   - No depende de descargas por internet ni APIs externas durante el flujo de trabajo.

2. **Detección del Punto Óptimo con Early Stopping y Paciencia Configurable:**
   - Evita tanto el **subentrenamiento** (*underfitting*) como el **sobreentrenamiento** (*overfitting*).
   - Monitorea una partición de validación (15%). Si la pérdida de validación deja de mejorar durante $N$ épocas consecutivas (paciencia por defecto: 3), detiene el entrenamiento de inmediato y **restaura automáticamente los pesos del mejor punto óptimo histórico**.

3. **Comando de Limpieza y Validaciones de Seguridad:**
   - Comando `clean` para resetear el entorno y reentrenar cuantas veces sea necesario sin alterar el dataset local.
   - **Validación estricta:** Bloquea el escaneo en vivo con la cámara (`live`) y la evaluación (`eval` / `compare`) si no se ha entrenado un modelo previamente, guiando al usuario con mensajes claros en consola.

4. **Red Neuronal Convolucional (CNN Simple y Robusta):**
   - Supera las limitaciones y complejidades de la regresión lineal utilizando detección espacial de trazos (filtros $3 \times 3$), submuestreo (`MaxPool2d`) e inferencia multiclase con **Softmax**.

5. **Fase Separada de Aumento de Datos (*Data Augmentation*):**
   - Agrupación por carpetas de clase (`digito_0` a `digito_9`) con 8 variaciones morfológicas por muestra: ruido gaussiano, rotaciones ($\pm 15^\circ$), dilatación (grueso), erosión (fino), deformación afín y volteado horizontal.
   - Posibilidad de entrenar y comparar el modelo estándar vs el modelo aumentado para evaluar el impacto en la precisión y robustez.

6. **Interfaz con Cámara Web y Botones en Pantalla:**
   - Además de los atajos de teclado, la ventana de OpenCV incluye **botones interactivos cliqueables con el ratón**:
     - `[ESPACIO] Capturar`: Congela el fotograma y clasifica el dígito.
     - `[A] Modo Auto`: Alterna el muestreo automático continuo.
     - `[Q] Salir`: Cierra la cámara limpiamente.
   - Área de foco (ROI) central y recuadro Picture-in-Picture (PiP) para visualizar la entrada normalizada $28 \times 28$.

---

## 2. Estructura del Proyecto

```
SUPERVISADO/
├── run.sh                          # Script ejecutable con el entorno virtual
├── requirements.txt                # Dependencias fijadas (torch, opencv, numpy, etc.)
├── models_saved/
│   ├── mnist_cnn_standard.pt       # Modelo CNN entrenado con datos estándar (Punto Óptimo)
│   └── mnist_cnn_augmented.pt      # Modelo CNN entrenado con Data Augmentation (Punto Óptimo)
├── data/
│   ├── mnist_local.npz             # Dataset MNIST 100% local (70,000 imágenes)
│   └── augmented_digits/           # Variaciones organizadas (digito_0/ ... digito_9/)
└── src/
    ├── config.py                   # Configuración inmutable (@dataclass)
    ├── models/
    │   ├── types.py                # Dataclasses inmutables y Type Aliases
    │   ├── interfaces.py           # Protocolos estructurales (IClassifierModel, IPreprocessor)
    │   ├── dataset.py              # Cargador 100% local (N, 1, 28, 28)
    │   ├── classifier.py           # SimpleCNN + EarlyStoppingTracker + CNNClassifier
    │   ├── augmenter.py            # Generador y agregador de variaciones morfológicas
    │   └── preprocessor.py         # Pipeline OpenCV: ROI -> Otsu -> Centrado 28x28
    ├── views/
    │   ├── cli_view.py             # Reportes en terminal, tabla de Early Stopping y comparativas
    │   └── camera_view.py          # Ventana OpenCV con ROI, PiP y Botones interactivos
    ├── controllers/
    │   ├── train_controller.py     # Entrenamiento con división Train/Val y Early Stopping
    │   ├── eval_controller.py      # Evaluación individual con validación de modelo
    │   ├── clean_controller.py     # Limpieza de modelos para recrear entrenamientos
    │   ├── augment_controller.py   # Fase de exportación de datos aumentados
    │   └── live_controller.py      # Bucle de cámara con validación de modelo y botones
    └── main.py                     # Punto de entrada CLI con subcomandos
```

---

## 3. Guía de Uso por Fases

Ejecuta cualquiera de las fases didácticas mediante `./run.sh`:

### Fase 0: Limpiar el Entrenamiento (Para recrearlo desde cero)
```bash
# Limpia solo los pesos de los modelos (.pt)
./run.sh src/main.py clean

# O limpia modelos y las imágenes aumentadas generadas
./run.sh src/main.py clean --augmented
```
*(El archivo `data/mnist_local.npz` nunca se borra, garantizando que no requieras conexión a internet).*

---

### Fase 1: Entrenamiento con Early Stopping y Paciencia Configurable

1. **Entrenar Modelo Estándar:**
   ```bash
   ./run.sh src/main.py train --epochs 15 --patience 3
   ```
2. **Entrenar Modelo con Data Augmentation:**
   ```bash
   ./run.sh src/main.py train --augmented --epochs 15 --patience 3
   ```

---

### Fase 2: Evaluación y Comparación Didáctica

1. **Evaluar el modelo restaurado:**
   ```bash
   ./run.sh src/main.py eval
   ```
   *(Si el modelo no está entrenado, se detiene y muestra la instrucción exacta).*

2. **Comparar ambos modelos lado a lado:**
   ```bash
   ./run.sh src/main.py compare
   ```

---

### Fase 3: Interacción en Tiempo Real con Cámara Web

Inicia la cámara web:
```bash
./run.sh src/main.py live
```
*(Si no has entrenado, no se abrirá la cámara y se mostrará una advertencia clara en consola solicitando entrenar primero).*

**Acciones disponibles en la ventana:**
- **Clic en `[ESPACIO] Capturar`:** Clasifica inmediatamente el dígito dentro del área de foco.
- **Clic en `[A] Modo Auto`:** Activa/Desactiva el muestreo continuo por intervalo.
- **Clic en `[Q] Salir`:** Finaliza la ejecución.
- *(Los atajos de teclado `ESPACIO`, `C`, `A` y `Q` también se mantienen activos).*
