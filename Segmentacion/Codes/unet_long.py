import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
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
ruta_base = '/home/tfd/mgutierr/tfm/Imagenes_juntas'

# Función para cargar y procesar las imágenes y sus etiquetas
def cargar_imagenes(ruta_json, etiquetas_unicas, tipo_imagen):
    X_imagenes = []
    Y_etiquetas = []

    with open(ruta_json, 'r') as file:
        datos = json.load(file)

    for entrada in datos:
        # Extraer el nombre del archivo de la ruta
        nombre_imagen = entrada['original_image'].split('/')[-1]  # Obtener solo el nombre del archivo
        imagen_path = os.path.join(ruta_base, nombre_imagen)  # Construir la ruta completa

        # Verificar si la etiqueta es 'bad' y omitir la imagen si es el caso
        calidad = entrada['Echography Quality']
        if calidad == 'bad':  # Si la etiqueta es 'bad', omitir esta entrada
            continue

        # Asegúrate de que la imagen existe
        if os.path.exists(imagen_path):
            imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)  # Carga en escala de grises
            # Verificar que la imagen se cargó correctamente
            if imagen is not None:
                # Redimensionar la imagen a 128x128
                imagen = cv2.resize(imagen, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                X_imagenes.append(imagen)
                Y_etiquetas.append(etiquetas_unicas[calidad])

    return X_imagenes, Y_etiquetas

# Cargar las imágenes longitudinales
X_imagenes_long, Y_etiquetas_long = cargar_imagenes('/home/tfd/mgutierr/tfm/DATA-Tunnel-Long/file_info.json', etiquetas_unicas, 'longitudinal')

# Función para cargar las roi
def cargar_mascaras(ruta_json, etiquetas_unicas, tipo_imagen):
    X_roi = []
    Y_roi = []

    with open(ruta_json, 'r') as file:
        datos = json.load(file)

    for entrada in datos:
        imagen_mask = entrada['drawing']
        etiqueta_mask = entrada['Echography Quality']

        # Si la etiqueta es 'bad', omitir esta entrada
        if etiqueta_mask == 'bad':
            continue

        # Construir la ruta completa del archivo de imagen
        imagen_mask_path = os.path.join(os.path.dirname(ruta_json), imagen_mask)

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

    return X_roi, Y_roi

# Cargar las máscaras longitudinales
X_roi_long, Y_roi_long = cargar_mascaras('/home/tfd/mgutierr/tfm/DATA-Tunnel-Long/file_info.json', etiquetas_unicas, 'longitudinal')

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
X_train, X_validacion, Y_train, Y_validacion, X_roi_train, X_roi_validacion = train_test_split(X_imagenes, X_masks_one_hot, X_roi, test_size=0.25, random_state=42)

print("Forma de X_train:", X_train.shape)
print("Forma de Y_train:", Y_train.shape)
print("Forma de X_validacion:", X_validacion.shape)
print("Forma de Y_validacion:", Y_validacion.shape)

# Build the model

inputs = tf.keras.layers.Input((256, 256, 1))
    # we have to compute the input values into floating values, that is normalizing the pixels in bt 0-1, if not it will interfere with 
    # the keras layers. That is from integers to floating values. 

s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
kernel_regularizer_encoder = tf.keras.regularizers.l2(0.001)
kernel_regularizer_decoder = tf.keras.regularizers.l2(0.001)

# Contraction path (encoder)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s) 
c1 = tf.keras.layers.Dropout(0.4)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1) 
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu',kernel_initializer='he_normal', padding='same') (p1)
c2 = tf.keras.layers.Dropout(0.4)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPool2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu',kernel_initializer='he_normal', padding='same') (p2)
c3 = tf.keras.layers.Dropout(0.4)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPool2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu',kernel_initializer='he_normal', padding='same') (p3)
c4 = tf.keras.layers.Dropout(0.4)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPool2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu',kernel_initializer='he_normal', padding='same') (p4)
c5 = tf.keras.layers.Dropout(0.4)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path (decoder)

u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.3)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.3)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.3)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1])
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.3)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(4, (1,1), activation='softmax')(c9)

n=23

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'), 
    tf.keras.callbacks.ModelCheckpoint(f'/home/tfd/mgutierr/tfm/U-net/Codigos/unet_long/mejor_modelo_long_{n}.h5', monitor='val_precision', save_best_only=True, mode='max', verbose=1)
    ] 

precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()

model = tf.keras.Model(inputs=[inputs], outputs = [outputs])
model.compile(optimizer='adam', loss=weighted_loss, metrics=['accuracy', precision_metric, recall_metric])

results = model.fit(
    X_train, Y_train,
    batch_size=35, 
    epochs=200, 
    validation_data=(X_validacion, Y_validacion), 
    callbacks=callbacks
    )


model.save(f'/home/tfd/mgutierr/tfm/U-net/Codigos/unet_long/modelo_unet_long_{n}.h5')

# Guardar la arquitectura
estructura_json = model.to_json()
with open(f"//home/tfd/mgutierr/tfm/U-net/Codigos/unet_long/estructura_modelo_{n}.json", "w") as json_file:
    json_file.write(estructura_json)


# Evaluar el modelo
loss, acc, precision, recall = model.evaluate(X_validacion, Y_validacion)

# Imprimir las métricas
print("Loss is = ", loss)
print("Accuracy is = ", acc * 100.0, "%")
print("Precision is = ", precision)
print("Recall is = ", recall)

#IOU
y_pred_val=model.predict(X_validacion)
y_pred_argmax_val=np.argmax(y_pred_val, axis=3) # con argmax buscas la clase de mayor probabilidad
IOU_keras = tf.keras.metrics.MeanIoU(num_classes=4)
Y_valid_argmax = np.argmax(Y_validacion, axis=-1)
print('Matriz de confusión:')
print(IOU_keras.update_state(Y_valid_argmax, y_pred_argmax_val))
confusion_matrix = IOU_keras.total_cm.numpy()
print('Confusion matrix acumulada:')
print(confusion_matrix)
print("Mean IoU on validation =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(4,4)
print(values)
class0_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class1_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class2_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class3_IoU = values[3,3] / (values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3] + values[1,3] + values[2,3])


print("IoU for class0 is: ", class0_IoU)
print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)


# Plot loss
plt.figure(figsize=(20, 15))
plt.subplot(2, 2, 1)
plt.plot(results.history['loss'], label='Training Loss')
plt.plot(results.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(2, 2, 2)
plt.plot(results.history['accuracy'], label='Training Accuracy')
plt.plot(results.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot precision
plt.subplot(2, 2, 3)
plt.plot(results.history['precision'], label='Training Precision')
plt.plot(results.history['val_precision'], label='Validation Precision')
plt.title('Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()

# Plot Recall
plt.subplot(2, 2, 4)
plt.plot(results.history['recall'], label='Training Recall')
plt.plot(results.history['val_recall'], label='Validation Recall')
plt.title('Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()

plt.savefig(f'/home/tfd/mgutierr/tfm/U-net/Graficas/unet_long/Graficas/Graph_{n}.png')


#IOU
y_pred_val=model.predict(X_validacion)
y_pred_argmax_val=np.argmax(y_pred_val, axis=3) # con argmax buscas la clase de mayor probabilidad
IOU_keras = tf.keras.metrics.MeanIoU(num_classes=4)
Y_valid_argmax = np.argmax(Y_validacion, axis=-1)
print('Matriz de confusión:')
print(IOU_keras.update_state(Y_valid_argmax, y_pred_argmax_val))
confusion_matrix = IOU_keras.total_cm.numpy()
print('Confusion matrix acumulada:')
print(confusion_matrix)
print("Mean IoU on validation =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(4,4)
print(values)
class0_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class1_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class2_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class3_IoU = values[3,3] / (values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3] + values[1,3] + values[2,3])

print("IoU for class0 is: ", class0_IoU)
print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)

# Calcular el F1-Score por clase
f1_scores = []
for i in range(4):  
    true_positive = confusion_matrix[i, i]
    false_positive = np.sum(confusion_matrix[:, i]) - true_positive
    false_negative = np.sum(confusion_matrix[i, :]) - true_positive
    
    # Evitar divisiones por cero
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    f1_scores.append(f1)
    print(f'F1-score for class {i} : ', f1)

# Crear la leyenda con los valores de IoU para cada clase
class_legend = [
    f'IoU for class 0: {class0_IoU:.4f}', 
    f'IoU for class 1: {class1_IoU:.4f}', 
    f'IoU for class 2: {class2_IoU:.4f}',  
    f'IoU for class 3: {class3_IoU:.4f}',
    f'F1-Score for class 0: {f1_scores[0]:.4f}', 
    f'F1-Score for class 1: {f1_scores[1]:.4f}', 
    f'F1-Score for class 2: {f1_scores[2]:.4f}',  
    f'F1-Score for class 3: {f1_scores[3]:.4f}'
]

# Visualizar la matriz de confusión con leyenda

# Crear la figura con espacio para la leyenda
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])  # 80% para la matriz, 20% para la leyenda

# Subgráfico para la matriz de confusión
ax1 = fig.add_subplot(gs[0])

sns.heatmap(confusion_matrix, 
            annot=True, 
            fmt="g", 
            cmap="Blues", 
            xticklabels=['Fondo', 'Nervio', 'Borde', 'Hueso'],
            yticklabels=['Fondo', 'Nervio', 'Borde', 'Hueso'], 
            cbar_kws={'label': 'IoU value'}, 
            ax=ax1, 
            vmax=180000)

ax1.set_title('Confusion Matrix with IoU and F1-Score by Class')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Subgráfico para la leyenda
ax2 = fig.add_subplot(gs[1])
ax2.axis('off')  # Desactivar el eje para que solo se vea la leyenda

# Añadir la leyenda en el segundo subgráfico
legend_text = '\n\n'.join(class_legend)  # Mayor separación entre líneas usando "\n\n"
ax2.text(0.3, 0.9, legend_text, 
         ha='left', va='top', fontsize=12, color='white', 
         bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=1'))

# Ajustar diseño antes de guardar
plt.tight_layout()

# Guardar la matriz de confusión como una imagen
plt.savefig(f'/home/tfd/mgutierr/tfm/U-net/Graficas/unet_long/CM/CM_{n}.png', bbox_inches='tight')


# Función para mostrar imagen y su máscara
def mostrar_imagen_y_mascara_con_roi(modelo, imagen, mascara_real, x_roi, indice):
   
    # Extraer una imagen, su máscara real y su ROI
    imagen_muestra = imagen[indice]
    mascara_real_muestra = mascara_real[indice]
    imagen_roi = x_roi[indice]

    # Obtener la máscara predicha por el modelo
    prediccion = modelo.predict(imagen[indice:indice + 1])
    mascara_predicha = np.argmax(prediccion[0], axis=-1)

    # Convertir la máscara real de one-hot a clases
    mascara_real_clases = np.argmax(mascara_real_muestra, axis=-1)

    # Normalizar las máscaras para visualización (0-255)
    mascara_predicha_normalizada = (mascara_predicha * 255 / (mascara_real.shape[-1] - 1)).astype(np.uint8)
    mascara_real_normalizada = (mascara_real_clases * 255 / (mascara_real.shape[-1] - 1)).astype(np.uint8)

    # Graficar
    plt.figure(figsize=(30, 15))
    
    # Imagen original
    plt.subplot(1, 4, 1)
    if imagen.shape[-1] == 1:  # Escala de grises
        plt.imshow(imagen_muestra.reshape(256, 256), cmap='gray')
    else:  # RGB
        plt.imshow(imagen_muestra.astype(np.uint8))
    plt.title('Imagen Original')
    plt.axis('off')

    # Imagen con ROI
    plt.subplot(1, 4, 2)
    if imagen_roi.shape[-1] == 1:  # Escala de grises
        plt.imshow(imagen_roi.reshape(128, 128), cmap='gray')
    else:  # RGB
        plt.imshow(imagen_roi.astype(np.uint8))
    plt.title('Imagen con ROI')
    plt.axis('off')

    # Máscara real
    plt.subplot(1, 4, 3)
    plt.imshow(mascara_real_normalizada, cmap='gray')
    plt.title('Máscara Real')
    plt.axis('off')

    # Máscara predicha
    plt.subplot(1, 4, 4)
    plt.imshow(mascara_predicha_normalizada, cmap='gray')
    plt.title('Máscara Predicha')
    plt.axis('off')


    # Guardar el resultado
    plt.savefig(f'/home/tfd/mgutierr/tfm/U-net/Graficas/unet_long/Segmentaciones/{n}_Segmentacion_{indice}.png', bbox_inches='tight')
    plt.close()

mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 10)
mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 11)
mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 22)
mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 33)
mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 44)
mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 55)
mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 66)
mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 77)
mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 88)
mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 99)
mostrar_imagen_y_mascara_con_roi(model, X_validacion, Y_validacion, X_roi_validacion, 100)