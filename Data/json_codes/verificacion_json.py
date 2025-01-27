import json

# Ruta del archivo JSON modificado (con diagnósticos añadidos)
ruta_json_modificado = '/home/tfd/mgutierr/tfm/Diagnostico/Long/mod_long.json'

# Leer el archivo con las entradas modificadas
with open(ruta_json_modificado, 'r') as file:
    json_modificado = json.load(file)

# Conjunto para almacenar los identificadores de pacientes con diagnóstico (diferenciando A y B)
pacientes_con_diagnostico = set()

# Contar los pacientes con diagnóstico
for entry in json_modificado:
    # Extraer el identificador del paciente y la mano (A o B) desde "original_image"
    paciente = entry['original_image'].split('/')[-1].split('_')[0]  # Ejemplo: "021BT_052.jpg" -> "021BT"
    paciente = paciente[:4]  # Obtener solo los primeros 4 caracteres, ejemplo "021B"
    
    # Verificar si la entrada tiene un diagnóstico
    if "Diagnostico" in entry:
        pacientes_con_diagnostico.add(paciente)  # Añadir el identificador completo del paciente (021B o 021A) al conjunto

# Ordenar los identificadores de pacientes por el número (los primeros 3 dígitos)
pacientes_ordenados = sorted(pacientes_con_diagnostico, key=lambda x: (int(x[:3]), x[3:]))

# El tamaño del conjunto es el número de pacientes con diagnóstico
numero_pacientes = len(pacientes_ordenados)

# Imprimir el número de pacientes con diagnóstico
print(f"Número de pacientes con diagnóstico: {numero_pacientes}")

# Imprimir los identificadores de los pacientes con diagnóstico en orden
print("Pacientes con diagnóstico (ordenados):")
for paciente in pacientes_ordenados:
    print(paciente)