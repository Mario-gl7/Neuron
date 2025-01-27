import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
import pickle
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


# Definir el mapeo de etiquetas a índices (one-hot encoding)
etiquetas_unicas = {'bad': 0, 'good': 1, 'fair': 2}  # Ajusta según tus clases
etiquetas_diagnostico = {'normal': 0, 'leve': 1, 'moderado': 2, 'severo': 3}

ruta_base = '/home/tfd/mgutierr/tfm/Imagenes_juntas'
ruta_json_nuevo = '/home/tfd/mgutierr/tfm/Diagnostico/Long/mod_long_depurado.json'
ruta_imagenes_roi = '/home/tfd/mgutierr/tfm/DATA-Tunnel-Long'

# Función para cargar y procesar las imágenes originales y sus etiquetas
def cargar_imagenes(ruta_json, etiquetas_unicas, etiquetas_diagnostico):
    X_imagenes = []
    Y_etiquetas = []
    Y_diag = []  # Array para almacenar los diagnósticos
    X_drawings = []
    X_nombres = []

    with open(ruta_json, 'r') as file:
        datos = json.load(file)

    for entrada in datos:
        # Extraer el nombre del archivo de la ruta
        nombre_imagen = entrada['original_image'].split('/')[-1]  # Obtener solo el nombre del archivo
        imagen_path = os.path.join(ruta_base, nombre_imagen)  # Construir la ruta completa
        drawing = entrada['drawing']

        # Verificar si la calidad es 'bad' y omitir la imagen si es el caso
        calidad = entrada['Echography Quality']
        if calidad == 'bad':  # Si la calidad es 'bad', omitir esta entrada
            continue

        # Agregar diagnóstico al array Y_diag
        diagnostico = entrada['Diagnostico']
        Y_diag.append(etiquetas_diagnostico[diagnostico])

        # Asegúrate de que la imagen existe
        if os.path.exists(imagen_path):
            imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)  # Carga en escala de grises
            # Verificar que la imagen se cargó correctamente
            if imagen is not None:
                # Redimensionar la imagen a 128x128
                imagen = cv2.resize(imagen, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                X_imagenes.append(imagen)
                Y_etiquetas.append(etiquetas_unicas[calidad])
                X_nombres.append(nombre_imagen)
                X_drawings.append(drawing)

    return X_imagenes, Y_etiquetas, Y_diag, X_drawings, X_nombres

X_imagenes_long, Y_etiquetas_long, Y_diag, X_drawings, X_nombres = cargar_imagenes(ruta_json_nuevo, etiquetas_unicas, etiquetas_diagnostico)

# Función para cargar las ROI y sus etiquetas
def cargar_mascaras(ruta_json, etiquetas_unicas):
    X_roi = []
    Y_roi = []
    Y_diag_roi = []  # Array para almacenar los diagnósticos

    with open(ruta_json, 'r') as file:
        datos = json.load(file)

    for entrada in datos:
        imagen_mask = entrada['drawing']
        etiqueta_mask = entrada['Echography Quality']

        # Si la calidad es 'bad', omitir esta entrada
        if etiqueta_mask == 'bad':
            continue

        # Agregar diagnóstico al array Y_diag_roi
        diagnostico = entrada['Diagnostico']
        Y_diag_roi.append(etiquetas_diagnostico[diagnostico])

        # Construir la ruta completa del archivo de imagen ROI
        imagen_mask_path = os.path.join(ruta_imagenes_roi, imagen_mask)

        # Asegúrate de que la imagen existe en el sistema de archivos antes de intentar leerla
        if os.path.exists(imagen_mask_path):
            # Leer la imagen en color (sin flag cv2.IMREAD_GRAYSCALE)
            imagen_mask = cv2.imread(imagen_mask_path)
            if imagen_mask is not None:
                # Redimensionar la imagen a 256x256
                imagen_mask = cv2.resize(imagen_mask, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                X_roi.append(imagen_mask)
                Y_roi.append(etiquetas_unicas[etiqueta_mask])
            else:
                print(f"Error al leer la imagen: {imagen_mask_path}")
        else:
            print(f"Imagen no encontrada: {imagen_mask_path}")

    return X_roi, Y_roi, Y_diag_roi

# Cargar las ROI transversales
X_roi_long, Y_roi_long, Y_diag_roi_long = cargar_mascaras(ruta_json_nuevo, etiquetas_unicas)

X_imagenes = X_imagenes_long 
Y_etiquetas = Y_etiquetas_long
X_roi = X_roi_long
Y_roi = Y_roi_long 

# Verificación 
assert len(X_imagenes) == len(X_roi)

for i in range(len(X_imagenes)):
    assert Y_etiquetas[i] == Y_roi[i], f"Desajuste en la imagen {i}: etiqueta y máscara no coinciden"

# Procesamiento de imágenes

X_imagenes = np.array(X_imagenes)
Y_etiquetas = np.array(Y_etiquetas)
X_roi = np.array(X_roi)
y_roi = np.array(Y_roi)

# Normalizamos los datos
X_imagenes = X_imagenes.reshape(X_imagenes.shape[0], 256, 256, 1)

X_roi = X_roi.reshape(X_roi.shape[0], 256, 256, 3)

# Definir los colores en formato RGB
color_nervio_bajo_rgb = [0-10, 255-10, 255-10]  # Rango inferior para nervio (RGB)
color_nervio_alto_rgb = [0+10, 255+10, 255+10]  # Rango superior para nervio (RGB)

color_borde_bajo_rgb = [255-10, 0-10, 255-10]  # Rango inferior para borde (RGB)
color_borde_alto_rgb = [255+10, 0+10, 255+10]  # Rango superior para borde (RGB)

color_ligamento_bajo_rgb = [250-10, 128-10, 114-10]  # Rango inferior para ligamento (RGB)
color_ligamento_alto_rgb = [250+10, 128+10, 114+10]  # Rango superior para ligamento (RGB)

color_hueso_bajo_rgb = [255-10, 255-10, 0-10]  # Rango inferior para hueso (RGB)
color_hueso_alto_rgb = [255+10, 255+10, 0+10]  # Rango superior para hueso (RGB)


# Inicializar las listas para almacenar las máscaras generadas
X_mask = []

# Función para crear la máscara de colores
def crear_mascara(color_bajo, color_alto, image_rgb):
    # Crear la máscara para los colores dentro del rango
    mask = np.all(np.logical_and(image_rgb >= color_bajo, image_rgb <= color_alto), axis=-1)
    return mask

# Iterar sobre todas las imágenes en x_roi
for image in X_roi:
    # Asegurarse de que la imagen no esté vacía
    if image is None:
        print("Error al cargar una imagen.")
        continue
    
    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Crear las máscaras para cada color
    mask_nervio = crear_mascara(color_nervio_bajo_rgb, color_nervio_alto_rgb, image_rgb)
    mask_borde = crear_mascara(color_borde_bajo_rgb, color_borde_alto_rgb, image_rgb)
    mask_hueso = crear_mascara(color_hueso_bajo_rgb, color_hueso_alto_rgb, image_rgb)

    # Crear una máscara combinada (array de un canal)
    image_combined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)  # Inicializamos con fondo (clase 0)
    # image_combined = np.full(image_rgb.shape[:2], -1, dtype=np.int8)  # Inicializamos con fondo (-1)

    # Asignar valores únicos para cada clase
    image_combined[mask_nervio] = 1       # Nervio (clase 1)
    image_combined[mask_borde] = 2        # Borde (clase 2)
    image_combined[mask_hueso] = 3        # Hueso (clase 4)

    # Añadir la máscara combinada a x_mask
    X_mask.append(image_combined)

# Convertir la lista x_mask a un array de numpy
X_mask = np.array(X_mask)

# Verificar que la cantidad de máscaras generadas coincida con las imágenes
assert len(X_mask) == len(X_imagenes), "El número de imágenes y máscaras no coincide"

# Ponderamos las clases
X_mask_flatt = X_mask.flatten() # las dimensiones de X_mask_flatt es: (12009472,)
y = X_mask_flatt
classes=np.unique(X_mask_flatt) # [0 1 2 3 4]
class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=X_mask_flatt) # Class weights are: [ 0.2065535  22.98925526 26.25135962 25.41983088 26.51946429]
class_weights_dict = {i: class_weights[i] for i in classes} # Class weights dictionary: {0: 0.20655350327821356, 1: 22.9892552570373, 2: 26.25135962227857, 3: 25.41983087978495, 4: 26.519464287685903}

X_masks_one_hot = to_categorical(X_mask, num_classes=4) # Estas son las etiquetas, que en vd son máscaras en formato one-hot

def weighted_loss(y_true, y_pred):
    # Convertir la máscara one-hot a índices de clases (es decir, la clase mayoritaria por píxel)
    class_indices = tf.argmax(y_true, axis=-1)  # Esto devuelve los índices de las clases para cada píxel
    
    # Obtener los pesos correspondientes a cada clase usando los índices
    class_weights = np.array([class_weights_dict.get(i, 1.0) for i in range(len(class_weights_dict))])
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
    
    # Obtener los pesos correspondientes a cada píxel utilizando los índices
    sample_weights = tf.gather(class_weights_tensor, class_indices)

    # Calcular la pérdida estándar de CategoricalCrossentropy
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(y_true, y_pred)

    # Multiplicar la pérdida por los pesos para cada píxel
    return loss * sample_weights

# Dividimos los datos
X_train, X_validacion, Y_train, Y_validacion, X_roi_train, X_roi_validacion = train_test_split(
    X_imagenes, X_masks_one_hot, X_roi, test_size=0.25, random_state=42)

z=23

model = tf.keras.models.load_model(f'/home/tfd/mgutierr/tfm/U-net/Codigos/unet_long/modelo_unet_long_{z}.h5', safe_mode=False, custom_objects={'weighted_loss': weighted_loss})


# Función para realizar la predicción y guardar las máscaras
def predecir_y_guardar(model, X_imagenes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mask_pred = []
    mask_nervios = []

    for i, imagen in enumerate(X_imagenes):
        
        imagen_input = np.expand_dims(imagen, axis=-1)  # Añadir la dimensión del canal
        imagen_input = np.expand_dims(imagen_input, axis=0)  # Añadir la dimensión del batch

        # Realizar la predicción
        prediccion = model.predict(imagen_input, verbose=0)

        # Convertir la predicción a una máscara con la clase de mayor probabilidad
        prediccion_clase = np.argmax(prediccion, axis=-1)  # Forma (1, 128, 128)
        prediccion_clase = np.squeeze(prediccion_clase)  # Eliminar dimensiones de tamaño 1
        mascara_clase_1 = np.uint8(prediccion_clase == 1) * 255 

        # Añadir la máscara a mask_pred
        mask_pred.append(prediccion_clase)
        mask_nervios.append(mascara_clase_1)

    return mask_nervios

# Definir el directorio de salida para guardar las predicciones
output_dir = '/home/tfd/mgutierr/tfm/predicciones_imagenes'

# Realizar las predicciones y guardarlas
mask = predecir_y_guardar(model, X_imagenes, output_dir)

mask_nervios = np.array(mask)
print('Shape X_imagenes: ', X_imagenes.shape)
print('Shape mask_nervios: ', mask_nervios.shape)
print('Nombres: ', len(X_nombres))
print('Drawings: ', len(X_drawings))

nervios = []
tamaño_fijo = 70 # Tamaño fijo del recorte

for i, (mascara, imagen) in enumerate(zip(mask_nervios, X_imagenes_long)):
    # Coordenadas del bounding box a partir de la máscara
    coords = np.column_stack(np.where(mascara == 255))  # Índices de píxeles de clase 1
    if coords.size > 0:
        # Bounding box inicial
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        # Centro del bounding box
        x_centro = (x_min + x_max) // 2
        y_centro = (y_min + y_max) // 2

        # Calcular los límites del nuevo bounding box (70x70)
        x_min_nuevo = max(0, x_centro - tamaño_fijo // 2)
        x_max_nuevo = min(imagen.shape[1], x_centro + tamaño_fijo // 2)
        y_min_nuevo = max(0, y_centro - tamaño_fijo // 2)
        y_max_nuevo = min(imagen.shape[0], y_centro + tamaño_fijo // 2)

        # Ajustar el tamaño para que sea exactamente 70x70 (si toca los bordes)
        if x_max_nuevo - x_min_nuevo < tamaño_fijo:
            if x_min_nuevo == 0:
                x_max_nuevo = tamaño_fijo
            elif x_max_nuevo == imagen.shape[1]:
                x_min_nuevo = imagen.shape[1] - tamaño_fijo

        if y_max_nuevo - y_min_nuevo < tamaño_fijo:
            if y_min_nuevo == 0:
                y_max_nuevo = tamaño_fijo
            elif y_max_nuevo == imagen.shape[0]:
                y_min_nuevo = imagen.shape[0] - tamaño_fijo

        # Recortar la región del nervio con el bounding box ajustado
        recorte = imagen[x_min_nuevo:x_max_nuevo, y_min_nuevo:y_max_nuevo]

        # Validar que el recorte es exactamente 70x70
        assert recorte.shape == (tamaño_fijo, tamaño_fijo), f"Error en la forma del recorte: {recorte.shape}"

        nervios.append(recorte)
    else:
        drawing = X_drawings[i]
        nombre = X_nombres[i]
        print(f"No se encontró nervio en la máscara {i}.")
        print(f"  Drawing asociado: {drawing}")
        print(f"  Nombre asociado: {nombre}")
        Y_diag.pop(i)

nervios = np.array(nervios)
print('Shape Nervios: ', nervios.shape)
print('Shape Y_diag: ', len(Y_diag))

# def guardar_imagenes_aleatorias_con_info(nervios, X_drawings, X_nombres, output_dir, num_imagenes=5):
#     # Crear el directorio de salida si no existe
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Seleccionar índices aleatorios
#     indices_aleatorios = random.sample(range(len(nervios)), num_imagenes)
    
#     for i, idx in enumerate(indices_aleatorios):
#         # Obtener la imagen y su información correspondiente
#         imagen = nervios[idx]
#         drawing = X_drawings[idx]
#         nombre = X_nombres[idx]
        
#         # Guardar la imagen en el directorio de salida
#         output_path = os.path.join(output_dir, f"nervio_long_random_{i+1}.png")
#         plt.imsave(output_path, imagen, cmap='gray')
        
#         # Mostrar información asociada
#         print(f"Imagen {i+1}:")
#         print(f"  Nombre: {nombre}")
#         print(f"  Drawing: {drawing}")
#         print(f"  Guardada en: {output_path}\n")

# output_dir = "/home/tfd/mgutierr/tfm/Diagnostico/Procesado/Long"
# guardar_imagenes_aleatorias_con_info(nervios, X_drawings, X_nombres, output_dir, num_imagenes=5)

# Normalizamos nervios
nervios = nervios.reshape(nervios.shape[0], 70, 70)
Y_diag = to_categorical(Y_diag, num_classes=len(etiquetas_diagnostico)) 

# Filtro de Contraste Adaptativo (CLAHE)
def filtro_contraste_adaptativo(X_trans, clip_limit=0.5, tile_grid_size=(2, 2)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # Asegurarse de que los valores estén en el rango de 0-255
    X_trans = np.array([clahe.apply(np.uint8(img)) for img in X_trans])  
    return X_trans  

# Filtrado Mediano
def aplicar_filtro_mediano(X_trans, kernel_size=3):
    X_trans = np.array([cv2.medianBlur(img, kernel_size) for img in X_trans])
    return X_trans  

# Realce adicional del contraste
def realce_adicional_contraste(X_trans, alpha_strong=1.5, beta_strong=-30):
    # Aplicar un realce fuerte de contraste
    X_trans = np.array([cv2.convertScaleAbs(img, alpha=alpha_strong, beta=beta_strong) for img in X_trans])
    return X_trans

# Aplicación del filtro mediano
X_mediano = aplicar_filtro_mediano(nervios)
print('Shape X_mediano: ', X_mediano.shape, X_mediano.dtype)

# Aplicar CLAHE
X_contraste = filtro_contraste_adaptativo(X_mediano)
print('Shape X_contraste: ', X_contraste.shape, X_contraste.dtype)

# Realce adicional del contraste
X_contraste_fuerte = realce_adicional_contraste(X_contraste)
print('Shape X_contraste_fuerte: ', X_contraste_fuerte.shape, X_contraste_fuerte.dtype)
X_final = X_contraste_fuerte.reshape(X_contraste_fuerte.shape[0], 70, 70, 1)

# def guardar_imagenes_aleatorias_con_info(nervios, X_drawings, X_nombres, output_dir, num_imagenes=5):
#     # Crear el directorio de salida si no existe
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Seleccionar índices aleatorios
#     indices_aleatorios = random.sample(range(len(nervios)), num_imagenes)
    
#     for i, idx in enumerate(indices_aleatorios):
#         # Obtener la imagen y su información correspondiente
#         imagen = nervios[idx]
#         drawing = X_drawings[idx]
#         nombre = X_nombres[idx]
        
#         # Guardar la imagen en el directorio de salida
#         output_path = os.path.join(output_dir, f"nervio_filtro_{i+1}.png")
#         plt.imsave(output_path, imagen, cmap='gray')
        
#         # Mostrar información asociada
#         print(f"Imagen {i+1}:")
#         print(f"  Nombre: {nombre}")
#         print(f"  Drawing: {drawing}")
#         print(f"  Guardada en: {output_path}\n")

# def guardar_progreso_filtros(nervios, X_mediano, X_contraste, X_contraste_fuerte, output_dir, num_imagenes=5):
#     # Crear el directorio de salida si no existe
#     os.makedirs(output_dir, exist_ok=True)

#     # Seleccionar índices aleatorios
#     indices_aleatorios = random.sample(range(len(nervios)), num_imagenes)

#     for i, idx in enumerate(indices_aleatorios):
#         # Crear una figura para cada imagen
#         fig, axes = plt.subplots(1, 4, figsize=(16, 6))

#         # Obtener las imágenes en las diferentes etapas
#         img_original = nervios[idx]
#         img_mediano = X_mediano[idx]
#         img_contraste = X_contraste[idx]
#         img_contraste_fuerte = X_contraste_fuerte[idx]

#         # Mostrar las imágenes
#         axes[0].imshow(img_original, cmap='gray')
#         axes[0].set_title("Original")
#         axes[0].axis('off')

#         axes[1].imshow(img_mediano, cmap='gray')
#         axes[1].set_title("Filtro Mediano")
#         axes[1].axis('off')

#         axes[2].imshow(img_contraste, cmap='gray')
#         axes[2].set_title("CLAHE")
#         axes[2].axis('off')

#         axes[3].imshow(img_contraste_fuerte, cmap='gray')
#         axes[3].set_title("Contraste Fuerte")
#         axes[3].axis('off')

#         # Ajustar la disposición de la figura
#         plt.tight_layout()

#         # Guardar la figura
#         output_path = os.path.join(output_dir, f"progreso_filtro_{i+1}.png")
#         plt.savefig(output_path, dpi=300)
#         print(f"Figura guardada en: {output_path}")

#         # Cerrar la figura para liberar memoria
#         plt.close(fig)

# output_dir = "/home/tfd/mgutierr/tfm/Diagnostico/Procesado/Long/Filtros"
# guardar_progreso_filtros(nervios, X_mediano, X_contraste, X_contraste_fuerte, output_dir, num_imagenes=5)
# guardar_imagenes_aleatorias_con_info(X_contraste_fuerte, X_drawings, X_nombres, output_dir, num_imagenes=5)

# Dividimos los datos

X_train_diag, X_validacion_diag, Y_train_diag, Y_validacion_diag = train_test_split(X_final, Y_diag, test_size=0.25, random_state=seed)

X_train_diag = np.array(X_train_diag, dtype=np.float32) / 255.0
X_validacion_diag = np.array(X_validacion_diag, dtype=np.float32) / 255.0
print('X_train_diag: ', X_train_diag.shape)
print('X_validacion_diag: ', X_validacion_diag.shape)

# Modelo de red

# modelo_1 = tf.keras.Sequential([

#     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(100, 100, 1)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Flatten(), # cambia de imagen cuadrada a vector simple

#     tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
#     tf.keras.layers.Dropout(0.4),
#     # tf.keras.layers.Dense(72, activation='relu', kernel_initializer='he_normal'),
#     # tf.keras.layers.Dropout(0.6),
#     # tf.keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal'),
#     tf.keras.layers.Dense(4, activation = 'softmax')
# ])

modelo = tf.keras.Sequential([

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', 
                           input_shape=(70, 70, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',  padding='same', kernel_initializer='he_normal'), 
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(4, activation = 'softmax')
])


modelo.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)

n=7

callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint(f'/home/tfd/mgutierr/tfm/Diagnostico/Codigos/Red/Modelos_diag_long/mejor_modelo_filtros_{n}.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
   ] 

print("Entrenando modelo...")
print('')

# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=20,        # Rotación aleatoria de las imágenes
    horizontal_flip=True,     # Volteo horizontal
    fill_mode='nearest'       # Modo de relleno para pixeles fuera del rango
)

datagen.fit(X_train_diag, seed=seed)

#Trabajar por lotes
tamaño_lote=20 
epocas=400

history = modelo.fit(
    datagen.flow(X_train_diag, Y_train_diag, 
    batch_size=tamaño_lote),
    epochs=epocas,
    validation_data=(X_validacion_diag, Y_validacion_diag),
    steps_per_epoch= int(np.ceil(len(X_train_diag) / float(tamaño_lote))),
    validation_steps= int(np.ceil(len(X_validacion_diag) / float(tamaño_lote))),
    callbacks=callbacks
)

print('')
print("Modelo entrenado!")
print('')


modelo.save(f'/home/tfd/mgutierr/tfm/Diagnostico/Codigos/Red/Modelos_diag_long/diag_long_filtros_{n}.keras')

# Guardar la arquitectura
estructura_json = modelo.to_json()
with open(f"/home/tfd/mgutierr/tfm/Diagnostico/Codigos/Red/Modelos_diag_long/estructura_modelo_filtros_{n}.json", "w") as json_file:
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

plt.savefig(f'/home/tfd/mgutierr/tfm/Diagnostico/Procesado/Long/Filtros/Modelos/{n}_Grafica_diag_long.png')

# Calculate the confusion matrix from the model's predictions and the ground truth
y_pred_val=modelo.predict(X_validacion_diag)
y_pred_argmax_val=np.argmax(y_pred_val, axis=1) # con argmax buscas la clase de mayor probabilidad
Y_valid_argmax = np.argmax(Y_validacion_diag, axis=-1)
conf_matrix = confusion_matrix(Y_valid_argmax.flatten(), y_pred_argmax_val.flatten(), labels=np.unique(Y_valid_argmax))

# Extraer el número de clases
num_clases = conf_matrix.shape[0]
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
mean_accuracy = class_accuracies.mean()
accuracy = metrics.accuracy_score(Y_valid_argmax.flatten(),y_pred_argmax_val.flatten())

# Preparar el texto de la leyenda
class_accuracies_text = "{ " + ", ".join([f'{i}={label}: {acc:.2f}' for i, (label, acc) in enumerate(zip(etiquetas_diagnostico.keys(), class_accuracies))]) + " }"
# metrics_text = (
#     f"Accuracy: {accuracy:.2f}\n"
#     f"Precisión por Clase: \n{class_accuracies_text}\n"
#     f"Mean Accuracy: {mean_accuracy:.2f}"
# )
legend = [
    f'Accuracy: {accuracy:.2f}', 
    f'Precisión por Clase: \n{class_accuracies_text}', 
    f'Mean Accuracy: {mean_accuracy:.2f}',  
]


# Graficar la matriz de confusión con seaborn
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[4, 1]) 

ax1 = fig.add_subplot(gs[0])

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y_valid_argmax), yticklabels=np.unique(Y_valid_argmax), ax=ax1)

# Configurar etiquetas y título
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')
ax1.set_title('Confusion Matrix')

# Subgráfico para la leyenda
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')  # Desactivar el eje para que solo se vea la leyenda

# Añadir la leyenda en el segundo subgráfico
legend_text = '\n'.join(legend)

ax2.text(0.5, 0.5, legend_text, 
         ha='center', va='center', fontsize=12, color='white', 
         bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=1'))

plt.tight_layout()

plt.savefig(f'/home/tfd/mgutierr/tfm/Diagnostico/Procesado/Long/Filtros/CM/{n}_CM_diag_long.png')

# Calcular y mostrar métricas
print("Accuracy = ", accuracy)
print(f"Precisión por Clase: {class_accuracies}")
print(f"Mean Accuracy: {mean_accuracy}")

tf.keras.backend.clear_session()