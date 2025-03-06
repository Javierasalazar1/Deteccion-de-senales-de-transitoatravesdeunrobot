# Importar las bibliotecas necesarias
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Cargar el modelo entrenado
model = tf.keras.models.load_model('Clasificacion_Trafic5_mejorado_con_null.h5')

# Diccionario para mapear las clases a nombres (ajusta según tus clases)
class_labels = {
    0: 'Clase 0',
    1: 'Clase 1',
    2: 'Clase 2',
    3: 'Clase 3',
    4: 'Clase 4'
}

# Uso del modelo con la cámara en tiempo real
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises y aplicar suavizado
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detección de bordes
    edges = cv2.Canny(blurred, 50, 150)

    # Encontrar contornos en la imagen
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar y encontrar el contorno más grande
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 500:  # Ajustar el umbral según lo necesario
            x, y, w, h = cv2.boundingRect(max_contour)
            roi = frame[y:y+h, x:x+w]

            # Preprocesar la región de interés (ROI)
            roi_resized = cv2.resize(roi, (224, 224))
            roi_array = np.array(roi_resized)
            roi_preprocessed = preprocess_input(roi_array)
            roi_expanded = np.expand_dims(roi_preprocessed, axis=0)

            # Hacer la predicción en la ROI
            preds = model.predict(roi_expanded)
            pred_class = np.argmax(preds, axis=1)[0]

            # Obtener el nombre de la clase predicha
            pred_class_name = class_labels.get(pred_class, 'Desconocido')

            # Dibujar el cuadro alrededor de la señal detectada y mostrar la clase predicha
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Clase: {pred_class_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el frame
    cv2.imshow('Detección de señales de tráfico', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()