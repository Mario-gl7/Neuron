import os
import json
import pandas as pd
import cv2

# Leer el archivo Excel
df = pd.read_excel('/home/tfd/mgutierr/tfm/diagnostico.xlsx')

# Función para obtener el diagnóstico para un paciente y mano específicos
def obtener_diagnostico(paciente, mano):
    # Filtramos la fila que corresponde al paciente y la mano
    fila = df[(df['PACIENTE'] == paciente) & (df['MANO'] == mano)]
    
    # Si encontramos la fila, obtenemos el diagnóstico
    if not fila.empty:
        diagnostico = fila['DIAGNOSTICO'].values[0]  # Tomamos el primer valor de diagnóstico
        return diagnostico
    else:
        return None  # En caso de no encontrar la fila

# Función para crear y guardar el JSON para todos los pacientes
def crear_y_guardar_json_todos_los_pacientes(nombre_archivo):
    # Crear una lista vacía para almacenar los datos de todos los pacientes
    todos_los_datos = []
    
    # Iterar sobre todos los pacientes y manos en el DataFrame
    for _, fila in df.iterrows():
        paciente = fila['PACIENTE']
        mano = fila['MANO']
        diagnostico = obtener_diagnostico(paciente, mano)
        
        if diagnostico is not None:
            # Crear un diccionario con los datos para el JSON
            data = {
                "PACIENTE": int(paciente),  # Convertir a tipo int
                "MANO": int(mano),          # Convertir a tipo int
                "DIAGNOSTICO": int(diagnostico)  # Convertir a tipo int
            }
            # Añadir el diccionario a la lista de datos
            todos_los_datos.append(data)
    
    # Convertir la lista de diccionarios a JSON
    json_data = json.dumps(todos_los_datos, indent=4)
    
    # Guardar el JSON en un archivo
    with open(nombre_archivo, 'w') as f:
        f.write(json_data)
    print(f"JSON guardado en {nombre_archivo}")

# Nombre del archivo donde se guardará el JSON
nombre_archivo = "/home/tfd/mgutierr/tfm/Diagnostico/entradas_excel.json"

# Crear y guardar el JSON para todos los pacientes
crear_y_guardar_json_todos_los_pacientes(nombre_archivo)
