import json

ruta_json = '/home/tfd/mgutierr/tfm/Diagnostico/entradas_excel.json'


def convertir_json(entrada):
    # Crear el nuevo identificador de paciente con 3 dígitos y sufijo B o A
    paciente_id = f"{entrada['PACIENTE']:03d}{'A' if entrada['MANO'] == 1 else 'B'}"
    
    # Crear el nuevo JSON con los campos deseados
    return {
        "paciente": paciente_id,
        "diagnostico": entrada["DIAGNOSTICO"]
    }

# Leer el archivo JSON de la ruta especificada
with open(ruta_json, 'r') as file:
    entradas = json.load(file)

# Convertir cada entrada en su formato correspondiente
resultados = [convertir_json(entrada) for entrada in entradas]

# Guardar el resultado en un nuevo archivo JSON
ruta_salida = '/home/tfd/mgutierr/tfm/Diagnostico/pacientes_converted.json'
with open(ruta_salida, 'w') as file:
    json.dump(resultados, file, indent=4)

print(f"Nuevo JSON guardado en: {ruta_salida}")