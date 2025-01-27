import json

# Ruta del archivo JSON modificado (con diagnósticos añadidos)
ruta_json_modificado = '/home/tfd/mgutierr/tfm/Diagnostico/Long/mod_long.json'

# Ruta para guardar el nuevo archivo con solo las entradas que tienen "Diagnostico"
ruta_salida = '/home/tfd/mgutierr/tfm/Diagnostico/Long/mod_long_depurado.json'

# Leer el archivo con las entradas modificadas
with open(ruta_json_modificado, 'r') as file:
    json_modificado = json.load(file)

# Filtrar las entradas que tienen el campo "Diagnostico"
entradas_con_diagnostico = [entry for entry in json_modificado if "Diagnostico" in entry]

# Guardar el nuevo archivo con solo las entradas que tienen "Diagnostico"
with open(ruta_salida, 'w') as file:
    json.dump(entradas_con_diagnostico, file, indent=4)

print(f"Nuevo archivo guardado con las entradas que tienen diagnóstico en: {ruta_salida}")