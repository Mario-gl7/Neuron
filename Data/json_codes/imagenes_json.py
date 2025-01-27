import os
import shutil
import json

# Rutas
ruta_imagenes = '/home/tfd/mgutierr/tfm/Imagenes_juntas'
ruta_json = '/home/tfd/mgutierr/tfm/Diagnostico/Long/mod_long_depurado.json'
ruta_destino = '/home/tfd/mgutierr/tfm/Diagnostico/Long'  # Carpeta de destino donde se copiarán las imágenes

# Leer el archivo JSON
with open(ruta_json, 'r') as file:
    datos_json = json.load(file)

# Crear la carpeta de destino si no existe
if not os.path.exists(ruta_destino):
    os.makedirs(ruta_destino)

# Contador de fotos copiadas
contador_fotos = 0

# Copiar las imágenes
for entry in datos_json:
    # Extraer el nombre de la imagen original
    nombre_imagen = entry['original_image'].split('/')[-1]  # Ejemplo: "023BT_016.jpg"
    
    # Ruta completa de la imagen en la carpeta de imágenes
    ruta_imagen_origen = os.path.join(ruta_imagenes, nombre_imagen)
    
    # Verificar si la imagen existe
    if os.path.exists(ruta_imagen_origen):
        # Copiar la imagen a la carpeta de destino
        ruta_imagen_destino = os.path.join(ruta_destino, nombre_imagen)
        shutil.copy(ruta_imagen_origen, ruta_imagen_destino)
        print(f"Imagen '{nombre_imagen}' copiada a la carpeta destino.")
        contador_fotos += 1  # Incrementar el contador de fotos copiadas
    else:
        print(f"La imagen '{nombre_imagen}' no se encuentra en la carpeta de imágenes.")

# Imprimir el total de fotos copiadas
print(f"Total de fotos copiadas: {contador_fotos}")

