import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import json
import cv2  # Para cargar imágenes
import datetime
import matplotlib.pyplot as plt
import random


# Fijar la semilla para que todo sea reproducible
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)  # Fija la semilla para los entornos de Python
random.seed(seed)  # Fija la semilla para la biblioteca random de Python
np.random.seed(seed)  # Fija la semilla para las operaciones de NumPy
tf.random.set_seed(seed)  # Fija la semilla para TensorFlow

# Inicializar listas para las imágenes y etiquetas
X_trans = []
Y_etiquetas = []

# Definir el mapeo de etiquetas a índices 
etiquetas_unicas = {'bad': 0, 'good': 1, 'fair': 2}  # Ajusta según tus clases

ruta_base = '/home/tfd/mgutierr/tfm/Imagenes_juntas'

# Imagenes Transversales

with open('/home/tfd/mgutierr/tfm/DATA-Tunnel-Trans/file_info.json', 'r') as file_trans:
    datos_trans = json.load(file_trans)

for entrada_l in datos_trans:
    # Extraer el nombre del archivo de la ruta
    nombre_imagen_trans = entrada_l['original_image'].split('/')[-1]  # Obtener solo el nombre del archivo
    imagen_path_trans = os.path.join(ruta_base, nombre_imagen_trans)  # Construir la ruta completa

    # Asegúrate de que la imagen existe
    if os.path.exists(imagen_path_trans):
        imagen_trans = cv2.imread(imagen_path_trans, cv2.IMREAD_GRAYSCALE)  # Carga en escala de grises
         # Verificar que la imagen se cargó correctamente
        if imagen_trans is not None:
            # Redimensionar la imagen 
            imagen_trans = cv2.resize(imagen_trans, (256, 256), interpolation=cv2.INTER_LANCZOS4)  # Ajusta a (ancho, alto)
            X_trans.append(imagen_trans)

            # Obtener la calidad y convertir a índice
            calidad_trans = entrada_l['Echography Quality']
            Y_etiquetas.append(etiquetas_unicas[calidad_trans])

# Normalización y procesamiento de imágenes
X_trans = np.array(X_trans)
Y_etiquetas = np.array(Y_etiquetas)
Y_etiquetas = Y_etiquetas.astype('float32')
print(X_trans.shape)
print(Y_etiquetas.shape)

X_trans = X_trans.reshape(X_trans.shape[0], 256, 256, 1)
X_trans = X_trans.astype('float32') / 255

Y_etiquetas = to_categorical(Y_etiquetas, num_classes=len(etiquetas_unicas)) 

# Dividimos los datos
X_train, X_validacion, Y_train, Y_validacion = train_test_split(X_trans, Y_etiquetas, test_size=0.25, random_state=42)

# modelo_1 = tf.keras.Sequential([

#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(256, 256, 1)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Conv2D(256, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
#     tf.keras.layers.Dense(3, activation = 'softmax')
# ])

modelo = tf.keras.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(256, 256, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(3, activation = 'softmax')
])

modelo.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)

n=23

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(f'/home/tfd/mgutierr/tfm/CNN/Codigos/Modelos_clasificacion_trans/{n}_mejor_clasificacion_trans.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
   ] 

# Crear un generador de data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,         # Rotación aleatoria en grados
    width_shift_range=0.1,     # Desplazamiento horizontal
    height_shift_range=0.1,    # Desplazamiento vertical
    zoom_range=0.1,            # Zoom aleatorio
    horizontal_flip=True,      # Inversión horizontal
    fill_mode='nearest'        # Relleno de píxeles al aplicar transformaciones
)

datagen.fit(X_train, seed=seed)

print("Entrenando modelo...")
print('')

#Trabajar por lotes
tamaño_lote=25 # modelo 8 tiene 10
epocas=150

history = modelo.fit(
    datagen.flow(X_train, Y_train, 
    batch_size=tamaño_lote),
    epochs=epocas,
    validation_data=(X_validacion, Y_validacion),
    steps_per_epoch= int(np.ceil(len(X_train) / float(tamaño_lote))),
    validation_steps= int(np.ceil(len(X_validacion) / float(tamaño_lote))),
    callbacks=callbacks
)

print('')
print("Modelo entrenado!")
print('')


modelo.save(f'/home/tfd/mgutierr/tfm/CNN/Codigos/Modelos_clasificacion_trans/clasificacion_trans_{n}.keras')

# Guardar la arquitectura
estructura_json = modelo.to_json()
with open(f"/home/tfd/mgutierr/tfm/CNN/Codigos/Modelos_clasificacion_trans/estructura_modelo_{n}.json", "w") as json_file:
    json_file.write(estructura_json)

# Acceder a los datos del historial de entrenamiento
historial = history.history

# Graficar la pérdida
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(historial['loss'], label='Pérdida de Entrenamiento')
plt.plot(historial['val_loss'], label='Pérdida de Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida por Época')

# Graficar la precisión (si tu modelo tiene la métrica de precisión)
plt.subplot(1, 2, 2)
plt.plot(historial['accuracy'], label='Precisión de Entrenamiento')
plt.plot(historial['val_accuracy'], label='Precisión de Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión por Época')

plt.savefig(f'/home/tfd/mgutierr/tfm/CNN/Graficas/Transversal/Clasificacion/Graficas/{n}_Grafica_trans.png')


# Calculate the confusion matrix from the model's predictions and the ground truth
y_pred_val=modelo.predict(X_validacion)
y_pred_argmax_val=np.argmax(y_pred_val, axis=1)
Y_valid_argmax = np.argmax(Y_validacion, axis=-1)
conf_matrix = confusion_matrix(Y_valid_argmax.flatten(), y_pred_argmax_val.flatten(), labels=np.unique(Y_valid_argmax))

# Extraer el número de clases
num_clases = conf_matrix.shape[0]
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
mean_accuracy = class_accuracies.mean()
accuracy = metrics.accuracy_score(Y_valid_argmax.flatten(),y_pred_argmax_val.flatten())

# Preparar el texto de la leyenda
class_accuracies_text = "{ " + ", ".join([f'{i}={label}: {acc:.2f}' for i, (label, acc) in enumerate(zip(etiquetas_unicas.keys(), class_accuracies))]) + " }"
metrics_text = (
    f"Accuracy: {accuracy:.2f}\n"
    f"Precisión por Clase: \n{class_accuracies_text}\n"
    f"Mean Accuracy: {mean_accuracy:.2f}"
)

# Graficar la matriz de confusión con seaborn
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y_valid_argmax), yticklabels=np.unique(Y_valid_argmax), ax=ax)

# Configurar etiquetas y título
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')

# Reservar espacio en la parte inferior
fig.subplots_adjust(bottom=0.3)

# Crear un eje adicional para la leyenda
legend_ax = fig.add_axes([0.05, 0.05, 0.4, 0.2])  # [izq, abajo, ancho, alto]
legend_ax.axis('off')  # Ocultar los ejes del subplot

# Añadir texto con recuadro
legend_ax.text(
    0.95, 0.5,  # Coordenadas dentro del eje adicional
    metrics_text,
    wrap=True,
    horizontalalignment='left',
    verticalalignment='center',
    fontsize=10,
    color='white',
    bbox=dict(facecolor='black', alpha=0.8, edgecolor='black', boxstyle='round,pad=1')  # Fondo y borde
)

plt.savefig(f'/home/tfd/mgutierr/tfm/CNN/Graficas/Transversal/Clasificacion/CM/{n}_CM_trans.png')

# Calcular y mostrar métricas
print("Accuracy = ", accuracy)
print(f"Precisión por Clase: {class_accuracies}")
print(f"Mean Accuracy: {mean_accuracy}")

tf.keras.backend.clear_session()