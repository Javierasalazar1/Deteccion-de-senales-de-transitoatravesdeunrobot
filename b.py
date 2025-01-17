# Importación de bibliotecas fundamentales
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import tensorflow.keras.backend as K
import gc
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configuración para usar memoria de GPU de forma incremental
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Definición del focal loss
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - \
               K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

# Configuración de variables
data = []
labels = []
classes = 5  # hay 5 clases, incluyendo la clase "nula"
image_size = 224  # Tamaño de las imágenes
cur_path = os.getcwd()

# Recuperación de las imágenes y sus etiquetas
for i in range(classes): 
    path = os.path.join(r'inserte la ruta del dataset', 'Train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((image_size, image_size))  # Redimensionamos las imágenes
            image = image.convert('RGB')  # Convertir la imagen a RGB
            image = np.array(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(i)  # Etiqueta para la clase correspondiente
        except:
            print(f"Error al cargar la imagen: {a}")

# Conversión de las listas a arrays de NumPy
data = np.array(data)
labels = np.array(labels)

# Verificar la distribución de clases
class_counts = Counter(labels)
print("\nDistribución de clases inicial:", class_counts)

# Calcular class weights para balanceo
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))
print("\nPesos de las clases:", class_weight_dict)

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

# Conversión a one-hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Configuración mejorada del data augmentation
datagen = ImageDataGenerator(
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,  # Desactivado para no confundir direcciones
    brightness_range=[0.8, 1.2],
    shear_range=0.15,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Modelo base
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(image_size, image_size, 3)
)

# Fine tuning más agresivo
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Arquitectura mejorada
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(classes, activation='softmax')(x)  

# Crear el modelo
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar con focal loss
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=focal_loss(),
    metrics=['accuracy']
)

# Callbacks mejorados
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model_with_null.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Parámetros de entrenamiento
epochs = 10
batch_size = 16

# Entrenamiento
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluación en conjunto de prueba
K.clear_session()
gc.collect()

# Cargar datos de prueba desde un archivo .csv
test_df = pd.read_csv(r'inserte la ruta del csv test')
labels_test = test_df["ClassId"].values
imgs = test_df["Path"].values

test_data = []
for img in imgs:
    image = Image.open(os.path.join(r'inserte la ruta de la carpeta test', img))
    image = image.resize((image_size, image_size))
    image = np.array(image)
    image = preprocess_input(image)
    test_data.append(image)

X_test_real = np.array(test_data)

# Cargar mejor modelo y hacer predicciones
model = load_model('best_model_with_null.h5', custom_objects={'focal_loss_fixed': focal_loss()})
pred = model.predict(X_test_real, batch_size=16)
pred_classes = np.argmax(pred, axis=-1)

# Análisis detallado de resultados
print("\nPrecisión en el conjunto de prueba:", accuracy_score(labels_test, pred_classes))
print("\nReporte de clasificación:")
print(classification_report(labels_test, pred_classes))
print("\nMatriz de confusión:")
conf_matrix = confusion_matrix(labels_test, pred_classes)
print(conf_matrix)

# Visualización de la matriz de confusión
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
tick_marks = np.arange(classes)
plt.xticks(tick_marks, ['Stop', '↑', '→', '←', 'Null'], rotation=45)
plt.yticks(tick_marks, ['Stop', '↑', '→', '←', 'Null'])
plt.xlabel('Predicción')
plt.ylabel('Verdadero')

# Anotar valores en la matriz
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# Guardar el modelo final
model.save('Clasificacion_Trafic5_mejorado_con_null.h5', include_optimizer=False)
