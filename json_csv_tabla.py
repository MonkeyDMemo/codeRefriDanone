import csv
import json

def json_a_csv_detallado(json_data, ruta_salida='conteo_productos.csv'):
    # 1. Extraer TODOS los productos únicos dinámicamente
    productos_unicos = set()
    for datos in json_data.values():
        productos_unicos.update(datos['conteo'].keys())  # Agrega todos los productos de cada imagen
    
    productos_unicos = sorted(productos_unicos)  # Orden alfabético

    # 2. Preparar encabezados
    encabezados = ['nombre_archivo', 'camara'] + list(productos_unicos)

    # 3. Escribir CSV eficientemente
    with open(ruta_salida, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(encabezados)  # Escribe encabezados
        
        for nombre_archivo, datos in json_data.items():
            # Construye fila: [nombre_archivo, camara, conteo_producto1, conteo_producto2...]
            fila = [nombre_archivo, datos['camara']]
            fila.extend(datos['conteo'].get(producto, 0) for producto in productos_unicos)
            writer.writerow(fila)
    
    print(f"✅ CSV guardado en: {ruta_salida}")
    print(f"📊 Productos detectados: {len(productos_unicos)}")
    print(f"📷 Imágenes procesadas: {len(json_data)}")

# Uso
with open(r'C:\Users\resendizjg\Downloads\resultados.json', 'r', encoding='utf-8') as f:
    datos_json = json.load(f)

json_a_csv_detallado(datos_json, r'C:\Users\resendizjg\Downloads\conteo_detallado.csv')