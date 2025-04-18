import cv2
import os
import time
import threading
import traceback
import logging
from datetime import datetime, timedelta
from ultralytics import YOLO
import json
import shutil
import csv
import pyodbc

# ----------------------------
# Configuraciones generales
# ----------------------------
PHOTO_INTERVAL = 10 * 60
DELETE_INTERVAL = 20 * 60
BURST_COUNT = 2
BURST_DELAY = 1
SAVE_FOLDER = "capturas"
RESULTADOS_FOLDER = "resultados_yolo"
IMG_SIZE = (960, 960)
APLICAR_TRANSFORMACION = True
CAMARAS_SELECCIONADAS = [0]
LOCK_FILE = ".captura.lock"
MODEL_PATH = r"C:\Users\resendizjg\Downloads\bestV5.pt"
GUARDAR_SEGMENTADAS = True
SUBIR_SQL = True  # Controla si se suben datos a la base de datos

# Configuración base de datos
DB_CONFIG = {
    'server': 'chatbotinventariosqlserver.database.windows.net',
    'database': 'Chabot_Inventario_Talento_SQLDB',
    'username': 'ghadmin',
    'password': 'wm5VrRK=jX/hE?-',
    'tabla': 'results_refrigerador_danone'
}

# ----------------------------
# Configuración de logging
# ----------------------------
LOG_FILE = "captura.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(RESULTADOS_FOLDER, exist_ok=True)

class CapturadorRafaga:
    def __init__(self, camaras, folder_base=SAVE_FOLDER):
        self.camaras = camaras
        self.folder_base = folder_base
        self.model = YOLO(MODEL_PATH)

    def verificar_camaras(self):
        disponibles = []
        for i in self.camaras:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                disponibles.append(i)
                cap.release()
            else:
                logging.warning(f"Cámara {i} no disponible")
        logging.info(f"Cámaras activas: {disponibles}")
        return disponibles

    def ejecutar_ronda(self, origen="manual"):
        if os.path.exists(LOCK_FILE):
            logging.warning(f"[{origen}] Ya hay una ejecución en curso. Abortando.")
            return
        open(LOCK_FILE, "w").close()

        try:
            camaras = self.verificar_camaras()
            ciclo_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_ciclo = os.path.join(self.folder_base, ciclo_timestamp)
            os.makedirs(folder_ciclo, exist_ok=True)
            logging.info(f"[{origen}] Iniciando ráfaga en carpeta: {folder_ciclo}")

            for burst_index in range(BURST_COUNT):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for cam_id in camaras:
                    try:
                        cap = cv2.VideoCapture(cam_id)
                        ret, frame = cap.read()
                        if ret:
                            if APLICAR_TRANSFORMACION:
                                frame = cv2.resize(frame, IMG_SIZE)
                            filename = f"cam{cam_id}_burst{burst_index}_{timestamp}.jpg"
                            path = os.path.join(folder_ciclo, filename)
                            cv2.imwrite(path, frame)
                            logging.debug(f"Foto guardada: {path}")
                        else:
                            logging.warning(f"[{origen}] No se pudo capturar imagen de cam{cam_id}")
                        cap.release()
                    except Exception as e:
                        logging.error(f"[{origen}] Error al capturar desde cam{cam_id}: {e}")
                        traceback.print_exc()
                time.sleep(BURST_DELAY)

            self.clasificar_imagenes_en_directorio(folder_ciclo, ciclo_timestamp)

        except Exception as e:
            logging.critical(f"[{origen}] Error crítico en ejecutar_ronda: {e}")
            traceback.print_exc()
        finally:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)

    def clasificar_imagenes_en_directorio(self, carpeta, nombre_directorio):
        try:
            resultados = {}
            imagenes = [f for f in os.listdir(carpeta) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for imagen in imagenes:
                ruta = os.path.join(carpeta, imagen)
                logging.info(f"Clasificando: {ruta}")
                res = self.model(ruta, conf=0.5, iou=0.6)

                conteo = {}
                for cls in res[0].boxes.cls.cpu().numpy():
                    label = self.model.names[int(cls)]
                    conteo[label] = conteo.get(label, 0) + 1

                resultados[imagen] = {
                    "camara": imagen.split("_")[0],
                    "timestamp_procesamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_captura": self.extraer_timestamp_desde_nombre(imagen),
                    "conteo": conteo
                }

                if GUARDAR_SEGMENTADAS:
                    res[0].save(filename=os.path.join(carpeta, f"segmentado_{imagen}"))

            json_path = os.path.join(RESULTADOS_FOLDER, f"{nombre_directorio}.json")
            with open(json_path, "w") as f:
                json.dump(resultados, f, indent=4)
            logging.info(f"Resultados guardados en {json_path}")

            self.generar_y_subir_csv(json_path, f"{RESULTADOS_FOLDER}/{nombre_directorio}.csv")

        except Exception as e:
            logging.error(f"Error en clasificar_imagenes_en_directorio: {e}")
            traceback.print_exc()

    def generar_y_subir_csv(self, json_path, csv_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            productos_unicos = set()
            for datos in json_data.values():
                productos_unicos.update(datos['conteo'].keys())
            productos_unicos = sorted(productos_unicos)

            encabezados = ['nombre_archivo', 'camara', 'timestamp_procesamiento', 'timestamp_captura'] + list(productos_unicos)

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
                    fila.extend(datos['conteo'].get(p, 0) for p in productos_unicos)
                    writer.writerow(fila)

            logging.info(f"CSV generado en: {csv_path}")

            # Subir a base de datos SQL Server
            config = DB_CONFIG
            connection_string = f"""
                Driver={{ODBC Driver 18 for SQL Server}};
                Server=tcp:{config['server']},1433;
                Database={config['database']};
                Uid={config['username']};
                Pwd={config['password']};
                Encrypt=yes;
                TrustServerCertificate=no;
                Connection Timeout=30;
            """

            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()
            tabla_completa = f"danone_proyecto.{config['tabla']}"

            cursor.execute(f"""
                SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = 'danone_proyecto' 
                AND TABLE_NAME = '{config['tabla']}'
            """)
            columnas_existentes = {row[0].lower() for row in cursor.fetchall()}

            if not columnas_existentes:
                columnas_sql = [f"{prod} INT DEFAULT 0" for prod in productos_unicos]
                create_table_sql = f"""
                    CREATE TABLE {tabla_completa} (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        nombre_archivo NVARCHAR(255) NOT NULL,
                        camara NVARCHAR(50) NOT NULL,
                        timestamp_procesamiento DATETIME NOT NULL,
                        timestamp_captura NVARCHAR(50) NOT NULL,
                        fecha_carga DATETIME DEFAULT GETDATE(),
                        {', '.join(columnas_sql)}
                    )
                """
                cursor.execute(create_table_sql)
                conn.commit()
                logging.info(f"Tabla {tabla_completa} creada")
            else:
                nuevos = [p for p in productos_unicos if p.lower() not in columnas_existentes]
                for p in nuevos:
                    try:
                        cursor.execute(f"ALTER TABLE {tabla_completa} ADD {p} INT DEFAULT 0")
                        logging.info(f"Columna añadida: {p}")
                    except Exception as e:
                        logging.warning(f"No se pudo añadir columna {p}: {str(e)}")
                if nuevos:
                    conn.commit()

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    columnas = []
                    valores = []
                    for campo in reader.fieldnames:
                        if campo.lower() in ['nombre_archivo', 'camara']:
                            columnas.append(campo)
                            valores.append(row[campo])
                        elif campo == 'timestamp_procesamiento':
                            columnas.append(campo)
                            valores.append(datetime.strptime(row[campo], '%Y-%m-%d %H:%M:%S'))
                        elif campo == 'timestamp_captura':
                            columnas.append(campo)
                            valores.append(row[campo])
                        elif campo in productos_unicos:
                            columnas.append(campo)
                            valores.append(int(row[campo]))

                    placeholders = ', '.join(['?'] * len(valores))
                    insert_sql = f"INSERT INTO {tabla_completa} ({', '.join(columnas)}) VALUES ({placeholders})"
                    try:
                        cursor.execute(insert_sql, valores)
                    except Exception as e:
                        logging.error(f"Error al insertar fila {row['nombre_archivo']}: {str(e)}")

            conn.commit()
            conn.close()
            logging.info(f"Datos insertados correctamente en {tabla_completa}")

        except Exception as e:
            logging.error(f"Error generando/subiendo CSV o cargando a SQL Server: {e}")
            traceback.print_exc()


    def extraer_timestamp_desde_nombre(self, nombre_archivo):
        try:
            partes = nombre_archivo.split("_")
            if len(partes) >= 3:
                return partes[2].replace(".jpg", "")
            return "desconocido"
        except:
            return "error"

    def eliminar_fotos_antiguas(self):
        try:
            ahora = datetime.now()
            for carpeta in os.listdir(self.folder_base):
                path_carpeta = os.path.join(self.folder_base, carpeta)
                if os.path.isdir(path_carpeta):
                    creado = datetime.fromtimestamp(os.path.getctime(path_carpeta))
                    if ahora - creado > timedelta(seconds=DELETE_INTERVAL):
                        try:
                            for archivo in os.listdir(path_carpeta):
                                os.remove(os.path.join(path_carpeta, archivo))
                            os.rmdir(path_carpeta)
                            logging.info(f"Carpeta eliminada: {path_carpeta}")
                        except Exception as e:
                            logging.error(f"Error al eliminar {path_carpeta}: {e}")
                            traceback.print_exc()
        except Exception as e:
            logging.critical(f"Error crítico en eliminar_fotos_antiguas: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", choices=["una_vez", "periodico"], default="una_vez")
    args = parser.parse_args()

    capturador = CapturadorRafaga(CAMARAS_SELECCIONADAS)

    if args.modo == "una_vez":
        logging.info("Modo: UNA VEZ")
        capturador.ejecutar_ronda(origen="una_vez")
        capturador.eliminar_fotos_antiguas()

    elif args.modo == "periodico":
        logging.info("Modo: PERIODICO")

        def loop_captura():
            while True:
                capturador.ejecutar_ronda(origen="periodico")
                capturador.eliminar_fotos_antiguas()
                time.sleep(PHOTO_INTERVAL)

        threading.Thread(target=loop_captura, daemon=True).start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Proceso terminado por el usuario.")
            cv2.destroyAllWindows()
            exit(0)
        except Exception as e:
            logging.critical(f"Error general en el hilo principal: {e}")
            traceback.print_exc()
            cv2.destroyAllWindows()
            exit(1)
        finally:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE) 