import csv
import json
from collections import defaultdict

def json_a_csv(json_data, ruta_salida='conteo_productos.csv'):
    # Extraer todos los productos únicos de todas las imágenes
    productos_unicos = set()
    for datos in json_data.values():
        for producto in datos['conteo'].keys():
            productos_unicos.add(producto)
    
    # Ordenar productos alfabéticamente
    productos_unicos = sorted(list(productos_unicos))
    
    # Agrupar por cámara
    conteo_por_camara = defaultdict(lambda: defaultdict(int))
    
    for nombre_archivo, datos in json_data.items():
        camara = datos['camara']
        for producto, cantidad in datos['conteo'].items():
            conteo_por_camara[camara][producto] += cantidad
    
    # Preparar encabezados
    encabezados = ['camara'] + productos_unicos
    
    # Escribir CSV
    with open(ruta_salida, 'w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)
        writer.writerow(encabezados)
        
        for camara, conteos in conteo_por_camara.items():
            fila = [camara]
            for producto in productos_unicos:
                fila.append(conteos.get(producto, 0))
            writer.writerow(fila)
    
    print(f"Archivo CSV creado: {ruta_salida}")

# Leer y parsear el archivo JSON
with open(r"C:\Users\resendizjg\Downloads\resultados.json", 'r') as f:
    json_data = json.load(f)

# Llamar a la función con los datos parseados
json_a_csv(json_data, r"C:\Users\resendizjg\Downloads\conteo_productos.csv")