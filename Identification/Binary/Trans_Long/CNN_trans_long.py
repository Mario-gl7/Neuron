import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
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
X_imagenes = []
Y_etiquetas = []

# Definir el mapeo de etiquetas a índices 
etiquetas_unicas = {'transversal': 0, 'longitudinal': 1}  

ruta_base = '/home/tfd/mgutierr/tfm/Imagenes_juntas'

# Imagenes transversales
with open('/home/tfd/mgutierr/tfm/DATA-Tunnel-Trans/file_info.json', 'r') as file_trans:
    datos_trans = json.load(file_trans)

for entrada in datos_trans:
    # Extraer el nombre del archivo de la ruta
    nombre_imagen_trans = entrada['original_image'].split('/')[-1]  # Obtener solo el nombre del archivo
    imagen_path_trans = os.path.join(ruta_base, nombre_imagen_trans)  # Construir la ruta completa

    if os.path.exists(imagen_path_trans):
        imagen_trans = cv2.imread(imagen_path_trans, cv2.IMREAD_GRAYSCALE)  # Carga en escala de grises
        
        if imagen_trans is not None:
           
            imagen_trans = cv2.resize(imagen_trans,(256, 256), interpolation=cv2.INTER_LANCZOS4)  # Ajusta a (ancho, alto)
            X_imagenes.append(imagen_trans)

            tipo_trans = entrada['Echography type']
            Y_etiquetas.append(etiquetas_unicas[tipo_trans])


# Imagenes Longitudinales

with open('/home/tfd/mgutierr/tfm/DATA-Tunnel-Long/file_info.json', 'r') as file_long:
    datos_long = json.load(file_long)

for entrada_l in datos_long:
    # Extraer el nombre del archivo de la ruta
    nombre_imagen_long = entrada_l['original_image'].split('/')[-1]  # Obtener solo el nombre del archivo
    imagen_path_long = os.path.join(ruta_base, nombre_imagen_long)  # Construir la ruta completa

    if os.path.exists(imagen_path_long):
        imagen_long = cv2.imread(imagen_path_long, cv2.IMREAD_GRAYSCALE)  # Carga en escala de grises
         
        if imagen_long is not None:
            
            imagen_long = cv2.resize(imagen_long, (256, 256), interpolation=cv2.INTER_LANCZOS4)  # Ajusta a (ancho, alto)
            X_imagenes.append(imagen_long)

            
            tipo_long = entrada_l['Echography type']
            Y_etiquetas.append(etiquetas_unicas[tipo_long])


# Normalización y procesamiento de imágenes
X_imagenes = np.array(X_imagenes)
Y_etiquetas = np.array(Y_etiquetas)
Y_etiquetas = Y_etiquetas.astype('float32')
print(X_imagenes.shape)
print(Y_etiquetas.shape)

X_imagenes = X_imagenes.reshape(X_imagenes.shape[0], 256, 256, 1)
X_imagenes = X_imagenes.astype('float32') / 255


# Dividimos los datos
X_train, X_validacion, Y_train, Y_validacion = train_test_split(X_imagenes, Y_etiquetas, test_size=0.20, random_state=seed)

X_train = X_train.astype('float32')
X_validacion = X_validacion.astype('float32')

Y_train = Y_train.astype('float32')
Y_validacion = Y_validacion.astype('float32')

print("Tipo de datos de X_train:", X_train.dtype)
print("Tipo de datos de Y_train:", Y_train.dtype)


# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=5,        # Rotación aleatoria de las imágenes
    zoom_range=0.1,           # Zoom aleatorio
    horizontal_flip=True,     # Volteo horizontal
    fill_mode='nearest'       # Modo de relleno para pixeles fuera del rango
)

datagen.fit(X_train, seed=seed)

# Red Neuronal

modelo_1 = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(256, 256, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(), 
    
    tf.keras.layers.Dense(256, activation='relu' , kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(85, activation='relu' , kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

modelo = tf.keras.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(256, 256, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(), 
    
    tf.keras.layers.Dense(256, activation='relu' , kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu' , kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu' , kernel_initializer='he_normal'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

modelo.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)

n=8

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(f'/home/tfd/mgutierr/tfm/CNN/Codigos/Modelos_CNN_trans_long/mejor_modelo_{n}.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
   ] 

print("Entrenando modelo...")
print('')

#Trabajar por lotes
tamaño_lote=40
epocas=100

history = modelo.fit(
    datagen.flow(X_train, Y_train, batch_size=tamaño_lote),
    epochs=epocas,
    validation_data=(X_validacion, Y_validacion),
    steps_per_epoch= int(np.ceil(len(X_train) / float(tamaño_lote))),
    validation_steps= int(np.ceil(len(X_validacion) / float(tamaño_lote))),
    callbacks=callbacks
)

print('')
print("Modelo entrenado!")
print('')

modelo.save(f'/home/tfd/mgutierr/tfm/CNN/Codigos/Modelos_CNN_trans_long/modelo_{n}.keras')

# Guardar la arquitectura
estructura_json = modelo.to_json()
with open(f"/home/tfd/mgutierr/tfm/CNN/Codigos/Modelos_CNN_trans_long/estructura_modelo_{n}.json", "w") as json_file:
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

# Graficar la precisión
plt.subplot(1, 2, 2)
plt.plot(historial['accuracy'], label='Precisión de Entrenamiento')
plt.plot(historial['val_accuracy'], label='Precisión de Validación')
mean_train_accuracy = sum(historial['accuracy']) / len(historial['accuracy'])
mean_val_accuracy = sum(historial['val_accuracy']) / len(historial['val_accuracy'])
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title('Precisión por Época')


plt.savefig(f'/home/tfd/mgutierr/tfm/CNN/Graficas/Trans_Long/{n}_modelo_translong.png')

# Generar predicciones en los datos de validación (1 canal)
y_pred_val = modelo.predict(X_validacion)
y_pred_argmax_val = (y_pred_val > 0.5).astype(int).flatten()
Y_valid_argmax = Y_validacion.flatten()

# Matriz de confusión
conf_matrix = confusion_matrix(Y_valid_argmax.flatten(), y_pred_argmax_val.flatten(), labels=np.unique(Y_valid_argmax))

# Extraer el número de clases
num_clases = conf_matrix.shape[0]
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
mean_accuracy = class_accuracies.mean()
accuracy = metrics.accuracy_score(Y_valid_argmax, y_pred_argmax_val)

# Graficar la matriz de confusión con seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y_valid_argmax), yticklabels=np.unique(Y_valid_argmax))

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
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')  # Fondo y borde
)

plt.savefig(f'/home/tfd/mgutierr/tfm/CNN/Graficas/Trans_Long/CM_{n}_translong.png')

# Calcular y mostrar métricas
print("Accuracy = ", accuracy)
print(f"Precisión por Clase: {class_accuracies}")
print(f"Mean Accuracy: {mean_accuracy}")

tf.keras.backend.clear_session()
