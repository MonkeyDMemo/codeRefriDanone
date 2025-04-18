import cv2
import os
import time
import threading
import traceback
import logging
import sys
import re
from datetime import datetime, timedelta
from ultralytics import YOLO
import json
import shutil
import csv
import pyodbc
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from dotenv import load_dotenv

# Configuración de logging (con soporte para UTF-8)
LOG_FILE = "captura.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Configuración de Azure Blob Storage
CONTAINER_NAME_IMAGES = "refrigeradordanoneimagenes"  # Cambiado a minúsculas sin caracteres especiales
CONTAINER_NAME_RESULTADOS = "refrigeradordanoneresultados"  # Cambiado a minúsculas sin caracteres especiales
load_dotenv()
AZURE_STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")

SUBIR_AZURE = True  # Bandera para activar/desactivar subida a Azure Blob

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
    'tabla': 'results_refrigerador_danone_iot_dev'
}

# Diccionario de IDs de productos
PRODUCT_ID_MAP = {
    "danup_envase": 1,
    "danonino": 2,
    "bonafont": 3,
    "danette_natilla": 4,
    "activia": 5,
    "objeto_no_identificado": 6,
    "dany_fresa": 7,
    "dany_gelatina": 8,
    "objects": 9
}

os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(RESULTADOS_FOLDER, exist_ok=True)

def sanitizar_nombre_archivo(nombre_archivo):
    """Sanitiza el nombre del archivo para cumplir con las restricciones de Azure Blob Storage"""
    # Convierte a minúsculas y reemplaza caracteres no permitidos
    nombre_limpio = re.sub(r'[^a-zA-Z0-9\-_.]', '', nombre_archivo).lower()
    return nombre_limpio

def subir_a_blob_storage(nombre_archivo, contenido_bytesio, container_name):
    try:
        # Sanitizar el nombre del archivo antes de subir
        nombre_sanitizado = sanitizar_nombre_archivo(nombre_archivo)
        
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=nombre_sanitizado)
        contenido_bytesio.seek(0)
        blob_client.upload_blob(contenido_bytesio, overwrite=True)
        url = f"https://chabotinventariostorage.blob.core.windows.net/{container_name}/{nombre_sanitizado}"
        logging.info(f"Archivo subido a Azure Blob Storage: {url}")
        return url
    except Exception as e:
        logging.error(f"Error al subir archivo a Blob Storage: {str(e)}")
        traceback.print_exc()
        return None

def subir_archivos_a_blob(json_path, imagenes_segmentadas_folder):
    if not SUBIR_AZURE:
        logging.info("Subida a Azure Blob Storage desactivada (SUBIR_AZURE = False)")
        return

    try:
        logging.info("Iniciando subida de archivos a Azure Blob Storage...")

        # Subir JSON de resultados
        nombre_json = os.path.basename(json_path)
        nombre_json_sanitizado = sanitizar_nombre_archivo(nombre_json)
        with open(json_path, "rb") as f:
            json_bytes = BytesIO(f.read())
            subir_a_blob_storage(nombre_json, json_bytes, CONTAINER_NAME_RESULTADOS)

        # Subir imágenes segmentadas
        segmentadas = [n for n in os.listdir(imagenes_segmentadas_folder) if n.startswith("segmentado_") and n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not segmentadas:
            logging.warning("No se encontraron imágenes segmentadas para subir")

        for nombre in segmentadas:
            full_path = os.path.join(imagenes_segmentadas_folder, nombre)
            with open(full_path, "rb") as f:
                imagen_bytes = BytesIO(f.read())
                subir_a_blob_storage(nombre, imagen_bytes, CONTAINER_NAME_IMAGES)

        logging.info("Subida a Azure Blob Storage finalizada exitosamente")

    except Exception as e:
        logging.error(f"Error en subida de archivos a Azure: {str(e)}")
        traceback.print_exc()

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
                            # Usar un formato de nombre más seguro para Azure
                            filename = sanitizar_nombre_archivo(f"cam{cam_id}_burst{burst_index}_{timestamp}.jpg")
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
                    # Nombre sanitizado para las imágenes segmentadas
                    segmentado_nombre = sanitizar_nombre_archivo(f"segmentado_{imagen}")
                    res[0].save(filename=os.path.join(carpeta, segmentado_nombre))

            # Nombre sanitizado para el JSON
            json_nombre = sanitizar_nombre_archivo(f"{nombre_directorio}.json")
            json_path = os.path.join(RESULTADOS_FOLDER, json_nombre)
            with open(json_path, "w") as f:
                json.dump(resultados, f, indent=4)
            logging.info(f"Resultados guardados en {json_path}")

            # Nombre sanitizado para el CSV
            csv_nombre = sanitizar_nombre_archivo(f"{nombre_directorio}.csv")
            csv_path = os.path.join(RESULTADOS_FOLDER, csv_nombre)
            self.generar_y_subir_csv(json_path, f"{RESULTADOS_FOLDER}/{nombre_directorio}.csv", nombre_directorio)
            logging.info(f"CSV generado en {csv_path}")

            # Subir JSON y segmentadas a Azure Blob Storage
            subir_archivos_a_blob(json_path, carpeta)

        except Exception as e:
            logging.error(f"Error en clasificar_imagenes_en_directorio: {e}")
            traceback.print_exc()

    def generar_y_subir_csv(self, json_path, csv_path, event_id):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Crear CSV plano con estructura solicitada
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['event_id', 'product_id', 'product_name', 'count', 'timestamp', 'imagen'])
                for nombre_archivo, datos in json_data.items():
                    timestamp_proc = datos['timestamp_procesamiento']
                    for producto, cantidad in datos['conteo'].items():
                        product_id = PRODUCT_ID_MAP.get(producto)
                        if product_id:
                            writer.writerow([event_id, product_id, producto, cantidad, timestamp_proc, nombre_archivo])
                        else:
                            logging.warning(f"⚠️ Producto no identificado (omitido): {producto}")

            logging.info(f"CSV generado en: {csv_path}")

            # Subir a SQL Server
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

            # Verificar existencia de tabla y crearla si no existe
            cursor.execute(f"""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                               WHERE TABLE_SCHEMA = 'danone_proyecto' AND TABLE_NAME = '{config['tabla']}')
                BEGIN
                    CREATE TABLE {tabla_completa} (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        event_id NVARCHAR(100) NOT NULL,
                        product_id INT NOT NULL,
                        product_name NVARCHAR(100) NOT NULL,
                        count INT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        imagen NVARCHAR(255) NOT NULL,
                        fecha_carga DATETIME DEFAULT GETDATE()
                    )
                END
            """)
            conn.commit()

            # Insertar registros desde CSV
            filas_insertadas = 0
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        cursor.execute(f"""
                            INSERT INTO {tabla_completa} (event_id, product_id, product_name, count, timestamp, imagen)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            row['event_id'],
                            int(row['product_id']),
                            row['product_name'],
                            int(row['count']),
                            datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S'),
                            row['imagen']
                        ))
                        filas_insertadas += 1
                    except Exception as e:
                        logging.error(f"❌ Error al insertar fila: {row} → {e}")

            conn.commit()
            conn.close()
            logging.info(f"✅ Total filas insertadas en {tabla_completa}: {filas_insertadas}")

        except Exception as e:
            logging.error(f"❌ Error en generar_y_subir_csv: {e}")
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