import json

ruta_json_modificado = '/home/tfd/mgutierr/tfm/Diagnostico/Long/file_info.json'
ruta_json = '/home/tfd/mgutierr/tfm/Diagnostico/pacientes_converted.json'

# Diccionario para mapear diagnóstico numérico a texto
diagnostico_map = {
    0: "normal",
    1: "leve",
    2: "moderado",
    3: "severo"
}

# Leer el archivo de pacientes y diagnóstico
with open(ruta_json, 'r') as file:
    diagnosticos = json.load(file)

# Leer el archivo con las entradas modificadas
with open(ruta_json_modificado, 'r') as file:
    json_modificado = json.load(file)

# Crear un diccionario de diagnóstico por paciente para acceso rápido
diagnosticos_dict = {entry['paciente']: entry['diagnostico'] for entry in diagnosticos}

# Añadir el diagnóstico al archivo modificado
for entry in json_modificado:
    # Extraer el identificador del paciente del nombre de archivo "original_image"
    paciente = entry['original_image'].split('/')[-1].split('_')[0]  # "021BT_052.jpg" -> "021BT"
    paciente = paciente[:4]  # Obtener solo los primeros 4 caracteres (ej. "021B")

    print(f"Procesando paciente: {paciente}")  # Mensaje de depuración

    if paciente in diagnosticos_dict:
        # Asignar el diagnóstico
        diagnostico_num = diagnosticos_dict[paciente]
        diagnostico_texto = diagnostico_map.get(diagnostico_num, "desconocido")
        entry["Diagnostico"] = diagnostico_texto
    else:
        print(f"Paciente {paciente} no encontrado en los diagnósticos.")  # Depuración si no se encuentra

# Guardar el archivo actualizado
ruta_salida = '/home/tfd/mgutierr/tfm/Diagnostico/Long/mod_long.json'
with open(ruta_salida, 'w') as file:
    json.dump(json_modificado, file, indent=4)

print(f"Archivo actualizado con diagnóstico guardado en: {ruta_salida}")