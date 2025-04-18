import csv
import json
import pyodbc
from datetime import datetime

def generar_y_subir_csv(json_path, csv_path, server, database, username, password, tabla):
    # 1. Leer y procesar JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Extraer TODOS los productos únicos
    productos_unicos = set()
    for datos in json_data.values():
        productos_unicos.update(datos['conteo'].keys())
    
    productos_unicos = sorted(productos_unicos)
    
    # 2. Generar CSV con estructura completa
    encabezados = [
        'nombre_archivo', 
        'camara',
        'timestamp_procesamiento',
        'timestamp_captura'
    ] + list(productos_unicos)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(encabezados)
        
        for nombre_archivo, datos in json_data.items():
            fila = [
                nombre_archivo,
                datos['camara'],
                datos['timestamp_procesamiento'],
                datos['timestamp_captura']
            ]
            fila.extend(datos['conteo'].get(producto, 0) for producto in productos_unicos)
            writer.writerow(fila)
    
    print(f"✅ CSV generado en: {csv_path} (Productos detectados: {len(productos_unicos)})")

    # 3. Conexión y sincronización con SQL Server
    try:
        connection_string = f"""
            Driver={{ODBC Driver 18 for SQL Server}};
            Server=tcp:{server},1433;
            Database={database};
            Uid={username};
            Pwd={password};
            Encrypt=yes;
            TrustServerCertificate=no;
            Connection Timeout=30;
        """
        
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        print("✅ Conexión a SQL Server establecida")

        # Nombre completo con esquema
        tabla_completa = f"danone_proyecto.{tabla}"
        
        # Verificar si la tabla existe
        cursor.execute(f"""
        SELECT COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'danone_proyecto' 
        AND TABLE_NAME = '{tabla}'
        """)
        
        columnas_existentes = {row[0].lower() for row in cursor.fetchall()}
        
        if not columnas_existentes:
            # Crear nueva tabla si no existe
            columns_sql = [f"{prod} INT DEFAULT 0" for prod in productos_unicos]
            create_table_sql = f"""
            CREATE TABLE {tabla_completa} (
                id INT IDENTITY(1,1) PRIMARY KEY,
                nombre_archivo NVARCHAR(255) NOT NULL,
                camara NVARCHAR(50) NOT NULL,
                timestamp_procesamiento DATETIME NOT NULL,
                timestamp_captura NVARCHAR(50) NOT NULL,
                fecha_carga DATETIME DEFAULT GETDATE(),
                {', '.join(columns_sql)}
            )
            """
            cursor.execute(create_table_sql)
            conn.commit()
            print(f"✅ Tabla creada con {len(productos_unicos)} columnas de productos")
        else:
            # Actualizar tabla existente con nuevos productos
            nuevos_productos = [p for p in productos_unicos if p.lower() not in columnas_existentes]
            
            if nuevos_productos:
                print(f"⚠️ Detectados {len(nuevos_productos)} nuevos productos")
                for producto in nuevos_productos:
                    try:
                        cursor.execute(f"""
                        ALTER TABLE {tabla_completa}
                        ADD {producto} INT DEFAULT 0
                        """)
                        print(f"  ➕ Columna añadida: {producto}")
                    except pyodbc.Error as e:
                        print(f"  ❌ Error al añadir columna {producto}: {str(e)}")
                        continue
                
                conn.commit()
                print(f"✅ Tabla actualizada con {len(nuevos_productos)} nuevas columnas")
        
        # 4. Insertar datos (con manejo de columnas dinámicas)
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Construir consulta dinámica
                columns = []
                values = []
                
                for field in reader.fieldnames:
                    if field.lower() in ['nombre_archivo', 'camara']:
                        columns.append(field)
                        values.append(row[field])
                    elif field == 'timestamp_procesamiento':
                        columns.append(field)
                        values.append(datetime.strptime(row[field], '%Y-%m-%d %H:%M:%S'))
                    elif field == 'timestamp_captura':
                        columns.append(field)
                        values.append(row[field])
                    elif field in productos_unicos:
                        columns.append(field)
                        values.append(int(row[field]))
                
                placeholders = ', '.join(['?'] * len(values))
                insert_sql = f"""
                INSERT INTO {tabla_completa} ({', '.join(columns)})
                VALUES ({placeholders})
                """
                
                try:
                    cursor.execute(insert_sql, values)
                except pyodbc.Error as e:
                    print(f"❌ Error al insertar registro {row['nombre_archivo']}: {str(e)}")
                    continue
        
        conn.commit()
        print(f"✅ Datos cargados exitosamente en {tabla_completa}")
        
    except pyodbc.Error as e:
        print(f"❌ Error de SQL Server:")
        print(f"Código: {e.args[0]}")
        print(f"Mensaje: {e.args[1]}")
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()
            print("Conexión cerrada")

# Configuración
config = {
    'json_path': r'C:\Users\resendizjg\Downloads\20250328_191536.json',
    'csv_path': r'C:\Users\resendizjg\Downloads\conteo_detallado.csv',
    'server': 'chatbotinventariosqlserver.database.windows.net',
    'database': 'Chabot_Inventario_Talento_SQLDB',
    'username': 'ghadmin',
    'password': 'wm5VrRK=jX/hE?-',
    'tabla': 'results_refrigerador_danone'
}

# Ejecutar
generar_y_subir_csv(**config)
# Generar CSV y subir a SQL Server
# generar_y_subir_csv( 