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
etiquetas_unicas = {'bad': 0, 'good': 1, 'fair': 1}

ruta_base = '/home/tfd/mgutierr/tfm/Imagenes_juntas'

# Imagenes transversales

with open('/home/tfd/mgutierr/tfm/DATA-Tunnel-Trans/file_info.json', 'r') as file_trans:
    datos_trans = json.load(file_trans)

for entrada in datos_trans:
    # Extraer el nombre del archivo de la ruta
    nombre_imagen_trans = entrada['original_image'].split('/')[-1]  # Obtener solo el nombre del archivo
    imagen_path_trans = os.path.join(ruta_base, nombre_imagen_trans)  # Construir la ruta completa

    # Asegúrate de que la imagen existe
    if os.path.exists(imagen_path_trans):
        imagen_trans = cv2.imread(imagen_path_trans, cv2.IMREAD_GRAYSCALE)  # Carga en escala de grises
         # Verificar que la imagen se cargó correctamente
        if imagen_trans is not None:
            # Redimensionar la imagen a 984x696
            imagen_trans = cv2.resize(imagen_trans,(256, 256), interpolation=cv2.INTER_LANCZOS4)  # Ajusta a (ancho, alto)
            X_imagenes.append(imagen_trans)

            # Obtener la calidad y convertir a índice
            tipo_trans = entrada['Echography Quality']
            Y_etiquetas.append(etiquetas_unicas[tipo_trans])


# Procesamiento de imágenes
X_imagenes = np.array(X_imagenes)
Y_etiquetas = np.array(Y_etiquetas)


# Normalizamos los datos
X_imagenes = X_imagenes.reshape(X_imagenes.shape[0], 256, 256, 1)

print('Shape X_imagenes: ', X_imagenes.shape, X_imagenes.dtype)
print('Shape Y_etiquetas: ', Y_etiquetas.shape, Y_etiquetas.dtype)

# Filtro de Contraste Adaptativo (CLAHE)
def filtro_contraste_adaptativo(X_trans, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    X_trans = np.array([clahe.apply(np.uint8(img * 255)) for img in X_trans])
    return X_trans[..., np.newaxis]  

# Filtrado Mediano
def aplicar_filtro_mediano(X_trans, kernel_size=3):
    X_trans = np.array([cv2.medianBlur(img, kernel_size) for img in X_trans])
    return X_trans[..., np.newaxis] 

# Umbralización de Otsu
def umbral_otsu(imagen):
    imagen_uint8 = np.uint8(imagen * 255) if imagen.dtype != np.uint8 else imagen
    _, imagen_umbralizada = cv2.threshold(imagen_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    return imagen_umbralizada

# Aplicar CLAHE
X_contraste = filtro_contraste_adaptativo(X_imagenes)
X_contraste = X_contraste.reshape(X_contraste.shape[0], 256, 256)
print('Shape X_contraste: ', X_contraste.shape, X_contraste.dtype)
print('Shape X_contraste: ', np.unique(X_contraste))

X_mediano = aplicar_filtro_mediano(X_contraste)
X_mediano = X_mediano.reshape(X_mediano.shape[0], 256, 256)
print('Shape X_mediano: ', X_mediano.shape, X_mediano.dtype)
print('Shape X_mediano: ', np.unique(X_mediano))

# Aplicar el método de Otsu a las imágenes
X_otsu = np.array([umbral_otsu(img) for img in X_mediano])
X_otsu = X_otsu.reshape(X_otsu.shape[0], 256, 256, 1) 
print('Shape X_otsu: ', X_otsu.shape, X_otsu.dtype)
print('Shape X_otsu: ', np.unique(X_otsu))

# Invertir las imágenes binarizadas
X_inv = np.array([cv2.bitwise_not(img) for img in X_otsu])
X_inv = X_inv.reshape(X_inv.shape[0], 256, 256, 1)
print('Shape X_inv: ', X_inv.shape, X_inv.dtype)
print('Shape X_inv: ', np.unique(X_inv))

X_final = X_imagenes * X_inv
print('Shape X_final: ', X_final.shape, X_final.dtype)
print('Shape X_final: ', np.unique(X_final))


# Función para visualizar el proceso
def visualizar_progresion(X_imagenes, X_contraste, X_mediano, X_otsu, X_inv, X_final, index=0):
    """
    Visualiza la progresión de la imagen original a través de los filtros aplicados y guarda la figura.
    """
    # Crear la figura con 6 subgráficas (una por cada paso)
    fig, axes = plt.subplots(1, 6, figsize=(20, 10))
    
    # Asegúrate de que las imágenes estén en formato adecuado para mostrar (en escala de grises)
    images = [X_imagenes[index], X_contraste[index], X_mediano[index], X_otsu[index], X_inv[index], X_final[index]]
    titles = ['Original', 'CLAHE', 'Filtro Mediano', 'Otsu', 'Inversión', 'Imagen Final']

    # Iterar sobre las subgráficas y mostrar cada imagen
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')  # Ocultar los ejes para mejor visualización

    # Guardar la figura con el índice en el nombre
    plt.tight_layout()
    plt.savefig(f'/home/tfd/mgutierr/tfm/CNN/Graficas/Transversal/Binario/Filtros/Fotos/progresion_filtros_{index}.png')  # Guardar la figura en formato PNG con el índice
    plt.show()

# Ejemplo de uso con el índice 0 a 3
for i in range(4):
    visualizar_progresion(X_imagenes, X_contraste, X_mediano, X_otsu, X_inv, X_final, index=i)


# Dividimos los datos
X_train, X_validacion, Y_train, Y_validacion = train_test_split(X_final, Y_etiquetas, test_size=0.25, random_state=seed)

X_train = np.array(X_train, dtype=np.float32) / 255.0
X_validacion = np.array(X_validacion, dtype=np.float32) / 255.0

print(f'Shape X_train: {X_train.shape}')
print(f'Shape X_validacion: {X_validacion.shape}')
print(f"Shape de y_train: {Y_train.shape}")
print(f"Shape de y_validacion: {Y_validacion.shape}")

modelo = tf.keras.Sequential([

    tf.keras.layers.Conv2D(8, (7, 7), activation='relu', padding='same', 
                           kernel_initializer='he_normal', input_shape=(256, 256, 1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same',
                            kernel_initializer='he_normal'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                           kernel_initializer='he_normal'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                           kernel_initializer='he_normal'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001), 
                          kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001),
                           kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001), 
                          kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

modelo.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)

n=7

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(f'/home/tfd/mgutierr/tfm/CNN/Codigos/Modelos_Binarios_trans/mejor_modelo_filtro_{n}.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
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

datagen.fit(X_train)


print("Entrenando modelo...")
print('')


#Trabajar por lotes
tamaño_lote=10 # 35, 15
epocas=300

history = modelo.fit(
    datagen.flow(X_train, Y_train, 
    batch_size=tamaño_lote),
    epochs=epocas,
    validation_data=(X_validacion, Y_validacion),
    steps_per_epoch= int(np.ceil(len(X_train) / float(tamaño_lote))),
    validation_steps= int(np.ceil(len(X_validacion) / float(tamaño_lote))),
    callbacks = callbacks
)

print('')
print("Modelo entrenado!")
print('')


modelo.save(f'/home/tfd/mgutierr/tfm/CNN/Codigos/Modelos_Binarios_trans/modelo_binario_filtro_{n}.keras')

# Guardar la arquitectura
estructura_json = modelo.to_json()
with open(f"/home/tfd/mgutierr/tfm/CNN/Codigos/Modelos_Binarios_trans/estructura_modelo_filtro_{n}.json", "w") as json_file:
    json_file.write(estructura_json)

# Acceder a los datos del historial de entrenamiento
historial = history.history

# Graficar la pérdida
plt.figure(figsize=(25, 5))
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
plt.text(
    len(historial['accuracy']) - 1,  # Posición en el eje x
    historial['val_accuracy'][-1] - 0.05,  # Posición en el eje y (ligeramente por debajo del último punto)
    f"Media Entrenamiento: {mean_train_accuracy:.2f}\nMedia Validación: {mean_val_accuracy:.2f}",
    fontsize=10,
    color='blue',
    bbox=dict(facecolor='white', alpha=0.5)  # Fondo del texto
)

plt.savefig(f'/home/tfd/mgutierr/tfm/CNN/Graficas/Transversal/Binario/Filtros/modelo_{n}_filtro_binario.png')

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

plt.savefig(f'/home/tfd/mgutierr/tfm/CNN/Graficas/Transversal/Binario/Filtros/CM_{n}_filtro_binario.png')

# Calcular y mostrar métricas
print("Accuracy = ", accuracy)
print(f"Precisión por Clase: {class_accuracies}")
print(f"Mean Accuracy: {mean_accuracy}")

tf.keras.backend.clear_session()
