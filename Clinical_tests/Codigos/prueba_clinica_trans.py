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

etiquetas_unicas = {'bad': 0, 'good': 1, 'fair': 1}

def leer_imagenes(input_dir):

    X_imagenes=[]
    # Leer todas las imágenes del directorio
    imagenes = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for i, ruta_imagen in enumerate(imagenes):
        # Leer la imagen en escala de grises
        imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen_original is None:
            print(f"No se pudo leer la imagen: {ruta_imagen}")
            continue

        # Redimensionar la imagen a (256, 256)
        imagen_resized = cv2.resize(imagen_original, (256, 256))
        X_imagenes.append(imagen_resized)
    
    return X_imagenes

# Directorios de entrada y salida
input_dir = '/home/tfd/mgutierr/tfm/rediagnostico/trans'

# Ejecutar la función
X_imagenes = leer_imagenes(input_dir)
X_imagenes = np.array(X_imagenes)
X_imagenes = X_imagenes.reshape(X_imagenes.shape[0], 256, 256, 1)

# Filtro de Contraste Adaptativo (CLAHE)
def filtro_contraste_adaptativo(X_trans, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    X_trans = np.array([clahe.apply(np.uint8(img * 255)) for img in X_trans])
    return X_trans[..., np.newaxis]  # Asegura que tenga un canal extra

# Filtrado Mediano
def aplicar_filtro_mediano(X_trans, kernel_size=3):
    X_trans = np.array([cv2.medianBlur(img, kernel_size) for img in X_trans])
    return X_trans[..., np.newaxis]  # Asegura que tenga un canal extra
    return np.array([cv2.medianBlur(img, kernel_size) for img in X_trans])

def umbralizar_imagen(imagen, umbral):
    """
    Aplica un umbral binario a una imagen.
    
    Args:
    - imagen: Imagen en escala de grises (valores entre 0 y 255).
    - umbral: Valor del umbral (por defecto 200).
    
    Returns:
    - Imagen umbralizada.
    """
    # Asegurarse de que la imagen sea de tipo uint8
    imagen_uint8 = np.uint8(imagen) if imagen.dtype != np.uint8 else imagen
    
    # Aplicar la umbralización
    _, imagen_umbralizada = cv2.threshold(imagen_uint8, umbral, 255, cv2.THRESH_BINARY)
    
    return imagen_umbralizada

# Función para aplicar umbralización de Otsu
def umbral_otsu(imagen):
    """
    Aplica el umbral de Otsu a una imagen en escala de grises.
    
    Args:
    - imagen: Imagen en escala de grises (valores entre 0 y 255).
    
    Returns:
    - Imagen binarizada usando el método de Otsu.
    """
    # Asegurar que la imagen esté en el rango correcto (0-255) y en uint8
    imagen_uint8 = np.uint8(imagen * 255) if imagen.dtype != np.uint8 else imagen
    
    # Aplicar el umbral de Otsu
    _, imagen_umbralizada = cv2.threshold(imagen_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return imagen_umbralizada

# Aplicar CLAHE
X_contraste = filtro_contraste_adaptativo(X_imagenes)
X_contraste = X_contraste.reshape(X_contraste.shape[0], 256, 256)
print('Shape X_contraste: ', X_contraste.shape, X_contraste.dtype)

X_mediano = aplicar_filtro_mediano(X_contraste)
X_mediano = X_mediano.reshape(X_mediano.shape[0], 256, 256)
print('Shape X_mediano: ', X_mediano.shape, X_mediano.dtype)

# X_umbral = umbralizar_imagen(X_mediano, umbral=60)
# X_umbral = X_umbral.reshape(X_umbral.shape[0], 256, 256, 1)
# print(f"Shape X_umbral: {X_umbral.shape}")

# Aplicar el método de Otsu a las imágenes
X_otsu = np.array([umbral_otsu(img) for img in X_mediano])
X_otsu = X_otsu.reshape(X_otsu.shape[0], 256, 256, 1)  # Asegurar que tenga el canal adicional
print('Shape X_otsu: ', X_otsu.shape, X_otsu.dtype)

# Invertir las imágenes binarizadas si es necesario
X_inv = np.array([cv2.bitwise_not(img) for img in X_otsu])
X_inv = X_inv.reshape(X_inv.shape[0], 256, 256, 1)
print('Shape X_inv: ', X_inv.shape, X_inv.dtype)

X_final = X_imagenes * X_inv
print('Shape X_final: ', X_final.shape, X_final.dtype)

X_final = np.array(X_final, dtype=np.float32) / 255.0

i=7

modelo_iden = tf.keras.models.load_model(f'/home/tfd/mgutierr/tfm/CNN/Codigos/Modelos_Binarios_trans/mejor_modelo_filtro_{i}.h5')

y_pred_val = modelo_iden.predict(X_final)
y_pred_argmax_val = (y_pred_val > 0.5).astype(int).flatten()


# Directorios para guardar las imágenes procesadas
output_dir_bad = "/home/tfd/mgutierr/tfm/Pruebas_clinicas/Identificación/Trans"
output_dir_good = "/home/tfd/mgutierr/tfm/Pruebas_clinicas/Identificación/Trans"

os.makedirs(output_dir_bad, exist_ok=True)
os.makedirs(output_dir_good, exist_ok=True)


# Función para agregar texto en la parte superior
def agregar_texto_encima(imagen, texto, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6  # Aumentar el tamaño de la fuente
    thickness = 1     

    # Calcular el tamaño del texto y ajustar la barra de fondo
    (text_width, text_height), baseline = cv2.getTextSize(texto, font, font_scale, thickness)
    barra_altura = text_height + baseline + 15  # Mayor espacio para la barra
    imagen_con_texto = cv2.copyMakeBorder(imagen, barra_altura, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Agregar la barra superior semi-transparente (blanco con opacidad)
    overlay = imagen_con_texto.copy()
    cv2.rectangle(overlay, (0, 0), (imagen.shape[1], barra_altura), (255, 255, 255), -1)  # Barra blanca
    alpha = 0.1  # Opacidad de la barra
    cv2.addWeighted(overlay, alpha, imagen_con_texto, 1 - alpha, 0, imagen_con_texto)

    # Agregar el texto en la barra superior
    text_x = 10
    text_y = barra_altura - 5
    cv2.putText(imagen_con_texto, texto, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return imagen_con_texto

# Función para procesar imágenes malas
def dibujar_cruz_con_texto(imagen, prediccion, precision):
    """Dibuja una cruz roja sobre la imagen y agrega texto rojo encima."""
    color_cruz = (0, 0, 255)  # Rojo en formato BGR
    color_texto = (0, 0, 255)  # Rojo para el texto
    thickness = 3

    # Dibujar la cruz roja
    height, width = imagen.shape[:2]
    imagen_con_cruz = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)  # Convertir a BGR
    cv2.line(imagen_con_cruz, (0, 0), (width - 1, height - 1), color_cruz, thickness)
    cv2.line(imagen_con_cruz, (0, height - 1), (width - 1, 0), color_cruz, thickness)

    # Agregar texto rojo encima
    texto = f"Pred: {prediccion}, Prec: {precision:.2f}"
    return agregar_texto_encima(imagen_con_cruz, texto, color_texto)

# Función para procesar imágenes buenas
def dibujar_recuadro_con_texto(imagen, prediccion, precision):

    color_recuadro = (0, 255, 0)  # Verde en formato BGR
    color_texto = (0, 255, 0)  # Verde para el texto
    thickness = 3

    # Dibujar el recuadro verde
    imagen_con_recuadro = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)  # Convertir a BGR
    cv2.rectangle(imagen_con_recuadro, (0, 0), (imagen.shape[1] - 1, imagen.shape[0] - 1), color_recuadro, thickness)

    # Agregar texto verde encima
    texto = f"Pred: {prediccion}, Prec: {precision:.2f}"
    return agregar_texto_encima(imagen_con_recuadro, texto, color_texto)

# Inicializar el array para las imágenes 'good'
X_good = []

# Evaluar predicciones y guardar imágenes
for i, (imagen_filtrada, imagen_original, pred, prob) in enumerate(zip(X_final, X_imagenes, y_pred_argmax_val, y_pred_val.flatten())):
    # Convertir la imagen original a uint8 para procesamiento
    imagen_original_uint8 = imagen_original.astype(np.uint8).reshape(256, 256)

    if pred == 0:  # Clase 'mala'
        # Dibujar cruz roja y texto sobre la imagen original
        imagen_tachada = dibujar_cruz_con_texto(imagen_original_uint8, "bad", prob)
        cv2.imwrite(os.path.join(output_dir_bad, f"imagen_{i}_bad.png"), imagen_tachada)
        # print(f"Imagen {i}: Predicción: 'bad', Precisión: {prob:.2f}")
    elif pred == 1:  # Clase 'buena'
        # Dibujar recuadro verde y texto sobre la imagen original
        imagen_recuadro = dibujar_recuadro_con_texto(imagen_original_uint8, "good", prob)
        cv2.imwrite(os.path.join(output_dir_good, f"imagen_{i}_good.png"), imagen_recuadro)
        # print(f"Imagen {i}: Predicción: 'good', Precisión: {prob:.2f}")

        # Agregar la imagen original (sin modificaciones) al array X_good
        X_good.append(imagen_original)


X_good = np.array(X_good)


class_weights_dict = {'0': 0.20655350327821356, '1': 22.9892552570373, '2': 26.25135962227857, '3': 25.41983087978495, '4': 26.519464287685903}

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

s=6

model_seg = tf.keras.models.load_model(f'/home/tfd/mgutierr/tfm/U-net/Codigos/unet_trans/modelo_unet_trans_{s}.h5', custom_objects={'weighted_loss': weighted_loss}, safe_mode=False)

def predecir_y_guardar(model, X_good, output_dir):
    # Asegúrate de que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)

    # Iterar sobre las imágenes originales (en este caso, las de X_good)
    for i, imagen_original in enumerate(X_good):

        # Normalizar la imagen y ajustar dimensiones
        imagen_input = np.expand_dims(imagen_original, axis=-1)  # Añadir canal
        imagen_input = np.expand_dims(imagen_input, axis=0)     # Añadir batch

        # Realizar la predicción
        prediccion = model.predict(imagen_input, verbose=0)

        # Convertir la predicción a una máscara con la clase de mayor probabilidad
        prediccion_clase = np.argmax(prediccion, axis=-1)  # Forma (1, 128, 128)
        prediccion_clase = np.squeeze(prediccion_clase)  # Eliminar dimensiones de tamaño 1
        prediccion_clase = np.uint8(prediccion_clase)

        # Crear la imagen con colores diferentes para cada clase
        mascara_coloreada = np.zeros((prediccion_clase.shape[0], prediccion_clase.shape[1], 3), dtype=np.uint8)

        # Colorear las clases
        mascara_coloreada[prediccion_clase == 0] = [0, 0, 0]  # Clase 0 en negro
        mascara_coloreada[prediccion_clase == 1] = [255, 255, 255]  # Clase 1 en blanco
        mascara_coloreada[prediccion_clase == 2] = [169, 169, 169]  # Clase 2 en gris
        mascara_coloreada[prediccion_clase == 3] = [255, 255, 255]  # Clase 3 en blanco
        mascara_coloreada[prediccion_clase == 4] = [255, 255, 255]  # Clase 4 en blanco

        # Crear una máscara binaria (blanco para la clase '1', negro para el resto)
        mascara_clase_1 = np.uint8(prediccion_clase == 1) * 255

        # Convertir la imagen original a BGR para poder mezclar con la máscara
        imagen_original_bgr = cv2.cvtColor(imagen_original.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Convertir la máscara a BGR para hacer la superposición
        # mascara_bgr = cv2.cvtColor(prediccion_clase, cv2.COLOR_GRAY2BGR)
        nervio_bgr = cv2.cvtColor(mascara_clase_1, cv2.COLOR_GRAY2BGR)

        # Aplicar transparencia a la imagen original (imagen original ligeramente transparente)
        alpha = 0.5  # Nivel de transparencia
        imagen_superpuesta = cv2.addWeighted(imagen_original_bgr, alpha, mascara_coloreada, 1 - alpha, 0)
        imagen_superpuesta_nervio = cv2.addWeighted(imagen_original_bgr, alpha, nervio_bgr, 1 - alpha, 0)

        # Guardar la figura combinada
        fig, ax = plt.subplots(1, 4, figsize=(15, 6))
        ax[0].imshow(imagen_original, cmap='gray')
        ax[0].set_title("Imagen Original")
        ax[0].axis('off')
        ax[1].imshow(prediccion_clase, cmap='gray')
        ax[1].set_title("Máscara Predicha")
        ax[1].axis('off')
        ax[2].imshow(imagen_superpuesta)
        ax[2].set_title("Imagen Superpuesta con Máscara")
        ax[2].axis('off')
        ax[3].imshow(imagen_superpuesta_nervio)
        ax[3].set_title("Imagen Superpuesta con Nervio")
        ax[3].axis('off')

        plt.tight_layout()

        # Definir el nombre de archivo de salida
        nombre_salida = os.path.join(output_dir, f"resultado_{i+1}.png")
        plt.savefig(nombre_salida)
        plt.close(fig)

        print(f"Resultado guardado en: {nombre_salida}")

# Directorios de salida
output_dir = '/home/tfd/mgutierr/tfm/Pruebas_clinicas/Segmentación/Trans'

# Ejecutar la función
predecir_y_guardar(model_seg, X_good, output_dir)

etiquetas_diagnostico = {'normal': 0, 'leve': 1, 'moderado': 2, 'severo': 3}

d=15

modelo_diag = tf.keras.models.load_model(f'/home/tfd/mgutierr/tfm/Diagnostico/Codigos/Red/Modelos_diag_trans/diag_trans_{d}.keras', safe_mode=False)


def procesar_y_visualizar_todas(model_seg, model_diag, X_good, etiquetas_diagnostico, output_dir):
    # Asegúrate de que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)

    for idx, imagen_original in enumerate(X_good):
        print(f"Procesando imagen {idx + 1}...")

        # Paso 1: Predicción de la máscara
        imagen_input = np.expand_dims(imagen_original, axis=(0, -1))  # Expandir a formato (1, 256, 256, 1)
        prediccion = model_seg.predict(imagen_input, verbose=0)
        mascara_predicha = np.argmax(prediccion, axis=-1).squeeze()

        # Paso 2: Extraer el bounding box del nervio
        coords = np.column_stack(np.where(mascara_predicha == 1))  # Clase del nervio
        if coords.size > 0:
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)

            # Ajustar el tamaño del recorte
            tamaño_fijo = 100
            x_centro, y_centro = (x_min + x_max) // 2, (y_min + y_max) // 2
            x_min_nuevo = max(0, x_centro - tamaño_fijo // 2)
            y_min_nuevo = max(0, y_centro - tamaño_fijo // 2)
            x_max_nuevo = min(imagen_original.shape[1], x_centro + tamaño_fijo // 2)
            y_max_nuevo = min(imagen_original.shape[0], y_centro + tamaño_fijo // 2)

            # Ajustar el tamaño para que sea exactamente 100x100 (si toca los bordes)
            if x_max_nuevo - x_min_nuevo < tamaño_fijo:
                if x_min_nuevo == 0:
                    x_max_nuevo = tamaño_fijo
                elif x_max_nuevo == imagen_original.shape[1]:
                    x_min_nuevo = imagen_original.shape[1] - tamaño_fijo

            if y_max_nuevo - y_min_nuevo < tamaño_fijo:
                if y_min_nuevo == 0:
                    y_max_nuevo = tamaño_fijo
                elif y_max_nuevo == imagen_original.shape[0]:
                    y_min_nuevo = imagen_original.shape[0] - tamaño_fijo

            recorte = imagen_original[x_min_nuevo:x_max_nuevo, y_min_nuevo:y_max_nuevo]

            # Paso 3: Predecir el estado del nervio
            recorte_normalizado = recorte / 255.0
            recorte_input = np.expand_dims(recorte_normalizado, axis=(0, -1))  # Expandir a formato (1, 100, 100, 1)
            predicciones = model_diag.predict(recorte_input, verbose=0).squeeze()
            estado_predicho_idx = np.argmax(predicciones)
            estado_predicho = [key for key, val in etiquetas_diagnostico.items() if val == estado_predicho_idx][0]
            precision_predicha = predicciones[estado_predicho_idx]

            # Ajustar las coordenadas del bounding box
            margen = 5  # Cantidad de píxeles para ampliar el bounding box
            x_min = max(0, x_min - margen)
            y_min = max(0, y_min - margen)
            x_max = min(imagen_original.shape[1], x_max + margen)
            y_max = min(imagen_original.shape[0], y_max + margen)

            # Paso 4: Anotar la imagen original
            imagen_annotada = cv2.cvtColor(imagen_original, cv2.COLOR_GRAY2BGR)  # Convertir a BGR para anotaciones
            cv2.rectangle(imagen_annotada, (y_min, x_min), (y_max, x_max), (255, 0, 0), 1)  # Dibujar el recuadro

            # Crear un fondo opaco para el texto
            texto = f"{estado_predicho} ({precision_predicha:.2f})"
            text_size = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = max(0, min(y_min, imagen_original.shape[1] - text_size[0] - 10))
            text_y = max(text_size[1] + 5, x_min - 10)
            text_bg_x1 = text_x
            text_bg_y1 = max(0, text_y - text_size[1] - 5)
            text_bg_x2 = text_x + text_size[0] + 10
            text_bg_y2 = text_y + 5

            cv2.rectangle(imagen_annotada, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (255, 0, 0), -1)  # Fondo azul opaco
            cv2.putText(imagen_annotada, texto, (text_x + 5, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)  # Superponer el diagnóstico y precisión
            
            # Guardar la imagen anotada
            nombre_salida = os.path.join(output_dir, f"{idx + 1}_diagnostico.png")
            cv2.imwrite(nombre_salida, imagen_annotada)
            print(f"Imagen guardada en {nombre_salida}")
        else:
            print(f"No se detectó nervio en la imagen {idx + 1}.")

# Directorio de salida
output_dir = '/home/tfd/mgutierr/tfm/Pruebas_clinicas/Diagnostico/Trans'

# Ejecutar para todas las imágenes
procesar_y_visualizar_todas(model_seg, modelo_diag, X_good, etiquetas_diagnostico, output_dir)