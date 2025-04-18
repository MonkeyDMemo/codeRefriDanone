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
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import argparse
import tqdm
from queue import Queue, Empty


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
CONTAINER_NAME_IMAGES = "refrigeradordanoneimagenes"
CONTAINER_NAME_RESULTADOS = "refrigeradordanoneresultados"
load_dotenv()
AZURE_STORAGE_CONNECTION_STRING = os.getenv("STORAGE_CONNECTION_STRING")

SUBIR_AZURE = True

# ----------------------------
# Definición de configuraciones por entornos
# ----------------------------
ENTORNOS = {
    "dev": {
        "CAMARAS_SELECCIONADAS": [0],
        "MODEL_PATH": r"C:\Users\resendizjg\Downloads\bestV5.pt",
        "TABLA_SQL": "results_refrigerador_danone_iot_dev",
        "PHOTO_INTERVAL": 10 * 60,
        "DELETE_INTERVAL": 20 * 60,
        "BURST_COUNT": 20,
        "BURST_DELAY": 1
    },
    "prod": {
        "CAMARAS_SELECCIONADAS": [2, 4],
        "MODEL_PATH": "/home/smart-cooler/Downloads/best.pt",
        "TABLA_SQL": "results_refrigerador_danone_iot_prod",
        "PHOTO_INTERVAL": 10 * 60,
        "DELETE_INTERVAL": 20 * 60,
        "BURST_COUNT": 20,
        "BURST_DELAY": 1
    }
}

# Configuraciones por defecto (se sobreescribirán según el entorno seleccionado)
PHOTO_INTERVAL = 10 * 60
DELETE_INTERVAL = 20 * 60
BURST_COUNT = 20
BURST_DELAY = 1
SAVE_FOLDER = "capturas"
RESULTADOS_FOLDER = "resultados_yolo"
IMG_SIZE = (960, 960)
APLICAR_TRANSFORMACION = True
CAMARAS_SELECCIONADAS = [0]
LOCK_FILE = ".captura.lock"
MODEL_PATH = r"C:\Users\resendizjg\Downloads\bestV5.pt"
GUARDAR_SEGMENTADAS = True
SUBIR_SQL = True
TABLA_SQL = "results_refrigerador_danone_iot_dev"

# Configuración base de datos
DB_CONFIG = {
    'server': 'chatbotinventariosqlserver.database.windows.net',
    'database': 'Chabot_Inventario_Talento_SQLDB',
    'username': 'ghadmin',
    'password': 'wm5VrRK=jX/hE?-',
    'tabla': TABLA_SQL
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

# Número máximo de trabajadores para procesamiento paralelo
MAX_WORKERS = 4

def aplicar_configuracion_entorno(entorno):
    """Aplica la configuración según el entorno especificado"""
    global CAMARAS_SELECCIONADAS, MODEL_PATH, TABLA_SQL, DB_CONFIG
    global PHOTO_INTERVAL, DELETE_INTERVAL, BURST_COUNT, BURST_DELAY
    
    if entorno not in ENTORNOS:
        logging.warning(f"Entorno '{entorno}' no reconocido. Usando configuración por defecto.")
        return
    
    config = ENTORNOS[entorno]
    
    # Aplicar configuraciones
    CAMARAS_SELECCIONADAS = config["CAMARAS_SELECCIONADAS"]
    MODEL_PATH = config["MODEL_PATH"]
    TABLA_SQL = config["TABLA_SQL"]
    PHOTO_INTERVAL = config["PHOTO_INTERVAL"]
    DELETE_INTERVAL = config["DELETE_INTERVAL"]
    BURST_COUNT = config["BURST_COUNT"]
    BURST_DELAY = config["BURST_DELAY"]
    
    # Actualizar la configuración de la base de datos
    DB_CONFIG["tabla"] = TABLA_SQL
    
    logging.info(f"Configuración aplicada para entorno: {entorno}")
    logging.info(f"Cámaras: {CAMARAS_SELECCIONADAS}")
    logging.info(f"Modelo: {MODEL_PATH}")
    logging.info(f"Tabla SQL: {TABLA_SQL}")

os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(RESULTADOS_FOLDER, exist_ok=True)

def sanitizar_nombre_archivo(nombre_archivo):
    """Sanitiza el nombre del archivo para cumplir con las restricciones de Azure Blob Storage"""
    nombre_limpio = re.sub(r'[^a-zA-Z0-9\-_.]', '', nombre_archivo).lower()
    return nombre_limpio

def subir_a_blob_storage(nombre_archivo, contenido_bytesio, container_name):
    try:
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

        total_files = len(segmentadas)
        with tqdm.tqdm(total=total_files, desc="Subiendo imágenes a Azure", unit="img") as pbar:
            for nombre in segmentadas:
                full_path = os.path.join(imagenes_segmentadas_folder, nombre)
                with open(full_path, "rb") as f:
                    imagen_bytes = BytesIO(f.read())
                    subir_a_blob_storage(nombre, imagen_bytes, CONTAINER_NAME_IMAGES)
                pbar.update(1)

        logging.info("Subida a Azure Blob Storage finalizada exitosamente")

    except Exception as e:
        logging.error(f"Error en subida de archivos a Azure: {str(e)}")
        traceback.print_exc()

class CapturadorRafaga:
    def __init__(self, camaras, folder_base=SAVE_FOLDER):
        self.camaras = camaras
        self.folder_base = folder_base
        self.model = YOLO(MODEL_PATH)
        # Cola para comunicación entre procesos
        self.cola_imagenes = Queue()
        # Resultados compartidos entre hilos
        self.resultados = {}
        self.resultados_lock = threading.Lock()
        # Executor para procesamiento paralelo
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        # Flag para detener los trabajadores
        self.detener_procesamiento = False
        # Contador de imágenes procesadas para barra de progreso
        self.imagenes_procesadas = 0
        self.total_imagenes = 0
        self.progreso_lock = threading.Lock()

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

    def extraer_timestamp_desde_nombre(self, nombre_archivo):
        try:
            partes = nombre_archivo.split("_")
            if len(partes) >= 3:
                return partes[2].replace(".jpg", "")
            return "desconocido"
        except:
            return "error"

    def clasificar_imagen(self, ruta_img, folder_ciclo, ciclo_timestamp, pbar=None):
        """Procesa una imagen individual con YOLO"""
        try:
            # Usar una instancia local del modelo para evitar problemas de concurrencia
            model = YOLO(MODEL_PATH)
            model.to("cpu")
            model.fuse()
            model.eval()
            
            res = model(ruta_img, conf=0.5, iou=0.6)
            conteo = {}
            for cls in res[0].boxes.cls.cpu().numpy():
                label = model.names[int(cls)]
                conteo[label] = conteo.get(label, 0) + 1

            if GUARDAR_SEGMENTADAS:
                res[0].save(filename=ruta_img.replace(os.path.basename(ruta_img), f"segmentado_{os.path.basename(ruta_img)}"))

            nombre_img = os.path.basename(ruta_img)
            
            with self.resultados_lock:
                self.resultados[nombre_img] = {
                    "camara": nombre_img.split("_")[0],
                    "timestamp_procesamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_captura": self.extraer_timestamp_desde_nombre(nombre_img),
                    "conteo": conteo
                }
            
            logging.info(f"📊 Clasificado (paralelo): {nombre_img} → {conteo}")
            
            # Actualizar barra de progreso si se proporciona
            with self.progreso_lock:
                self.imagenes_procesadas += 1
                if pbar is not None:
                    pbar.update(1)
            
            # Si es la última imagen a procesar, guardar resultados
            self.verificar_y_guardar_resultados(folder_ciclo, ciclo_timestamp)
            
        except Exception as e:
            logging.error(f"❌ Error clasificando imagen {ruta_img}: {e}")
            traceback.print_exc()
            # Actualizar barra de progreso incluso si hay error
            with self.progreso_lock:
                self.imagenes_procesadas += 1
                if pbar is not None:
                    pbar.update(1)

    def procesador_imagenes(self, folder_ciclo, ciclo_timestamp, pbar=None):
        """Función que se ejecuta en un hilo separado para procesar imágenes de la cola"""
        logging.info(f"🔄 Iniciando procesador de imágenes en hilo separado")
        
        while not self.detener_procesamiento:
            try:
                # Obtener una imagen de la cola (espera máximo 1 segundo)
                try:
                    ruta_img = self.cola_imagenes.get(timeout=1)
                    # Procesar la imagen en un hilo del pool
                    self.executor.submit(self.clasificar_imagen, ruta_img, folder_ciclo, ciclo_timestamp, pbar)
                except Empty:
                    # La cola está vacía, simplemente continuar
                    continue
                except Exception as e:
                    # Otro error, loggear y continuar
                    if not self.detener_procesamiento:
                        logging.error(f"Error al obtener o procesar imagen de la cola: {e}")
                    continue
                    
            except Exception as e:
                if not self.detener_procesamiento:
                    logging.error(f"❌ Error en procesador_imagenes: {e}")
                    traceback.print_exc()
        
        logging.info("🛑 Procesador de imágenes finalizado")

    def verificar_y_guardar_resultados(self, folder_ciclo, ciclo_timestamp):
        """Verifica si ya se procesaron todas las imágenes y guarda los resultados"""
        with self.resultados_lock, self.progreso_lock:
            # Contar imágenes en el directorio (excluyendo las segmentadas)
            imagenes = [f for f in os.listdir(folder_ciclo) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith("segmentado_")]
            
            # Si el tamaño de resultados es igual al número de imágenes y todas las imágenes esperadas fueron procesadas
            if len(self.resultados) == len(imagenes) and self.imagenes_procesadas >= self.total_imagenes and len(imagenes) > 0:
                json_path = os.path.join(RESULTADOS_FOLDER, f"{ciclo_timestamp}.json")
                
                # Verificar si ya existe el archivo para evitar escribir múltiples veces
                if not os.path.exists(json_path):
                    with open(json_path, "w") as f:
                        json.dump(self.resultados, f, indent=4)
                    logging.info(f"💾 Resultados guardados en {json_path}")
                    
                    csv_path = os.path.join(RESULTADOS_FOLDER, f"{ciclo_timestamp}.csv")
                    self.generar_y_subir_csv(json_path, csv_path, ciclo_timestamp)
                    
                    subir_archivos_a_blob(json_path, folder_ciclo)

    def ejecutar_ronda(self, origen="manual", mostrar_progreso=False):
        if os.path.exists(LOCK_FILE):
            logging.warning(f"[{origen}] Ya hay una ejecución en curso. Abortando.")
            return
        open(LOCK_FILE, "w").close()

        # Reiniciar contadores para la barra de progreso
        with self.progreso_lock:
            self.imagenes_procesadas = 0
            self.total_imagenes = BURST_COUNT * len(self.camaras)

        try:
            camaras = self.verificar_camaras()
            ciclo_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_ciclo = os.path.join(self.folder_base, ciclo_timestamp)
            os.makedirs(folder_ciclo, exist_ok=True)
            logging.info(f"[{origen}] Iniciando ráfaga en carpeta: {folder_ciclo}")
            
            # Reiniciar resultados para este ciclo
            with self.resultados_lock:
                self.resultados = {}
            
            # Actualizar total de imágenes esperadas
            with self.progreso_lock:
                self.total_imagenes = BURST_COUNT * len(camaras)
            
            # Crear barra de progreso si se solicita
            pbar = None
            if mostrar_progreso:
                pbar = tqdm.tqdm(total=self.total_imagenes, desc="Procesando imágenes", unit="img")
            
            # Iniciar hilo procesador de imágenes
            self.detener_procesamiento = False
            procesador_thread = threading.Thread(
                target=self.procesador_imagenes, 
                args=(folder_ciclo, ciclo_timestamp, pbar)
            )
            procesador_thread.daemon = True
            procesador_thread.start()

            # Barra de progreso para captura de imágenes
            captura_pbar = None
            if mostrar_progreso:
                captura_pbar = tqdm.tqdm(total=BURST_COUNT * len(camaras), desc="Capturando imágenes", unit="img")

            # Capturar imágenes y agregarlas a la cola de procesamiento
            for burst_index in range(BURST_COUNT):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for cam_id in camaras:
                    try:
                        cap = cv2.VideoCapture(cam_id)
                        ret, frame = cap.read()
                        if ret:
                            if APLICAR_TRANSFORMACION:
                                frame = cv2.resize(frame, IMG_SIZE)
                                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            
                            filename = sanitizar_nombre_archivo(f"cam{cam_id}_burst{burst_index}_{timestamp}.jpg")
                            path = os.path.join(folder_ciclo, filename)
                            cv2.imwrite(path, frame)
                            logging.debug(f"Foto guardada: {path}")
                            
                            # Agregar imagen a la cola de procesamiento
                            self.cola_imagenes.put(path)
                            
                            # Actualizar barra de progreso de captura
                            if mostrar_progreso and captura_pbar:
                                captura_pbar.update(1)
                        else:
                            logging.warning(f"[{origen}] No se pudo capturar imagen de cam{cam_id}")
                            # Actualizar contador de imágenes esperadas
                            with self.progreso_lock:
                                self.total_imagenes -= 1
                                if pbar:
                                    pbar.total = self.total_imagenes
                                    pbar.refresh()
                        cap.release()
                    except Exception as e:
                        logging.error(f"[{origen}] Error al capturar desde cam{cam_id}: {e}")
                        traceback.print_exc()
                        # Actualizar contador de imágenes esperadas
                        with self.progreso_lock:
                            self.total_imagenes -= 1
                            if pbar:
                                pbar.total = self.total_imagenes
                                pbar.refresh()
                time.sleep(BURST_DELAY)
            
            # Cerrar barra de progreso de captura
            if mostrar_progreso and captura_pbar:
                captura_pbar.close()
            
            # Esperar a que se procesen todas las imágenes
            while not self.cola_imagenes.empty() or self.imagenes_procesadas < self.total_imagenes:
                logging.info(f"Esperando a que se procesen las imágenes... ({self.imagenes_procesadas}/{self.total_imagenes})")
                time.sleep(2)
            
            # Señalizar finalización al hilo procesador
            self.detener_procesamiento = True
            procesador_thread.join(timeout=10)
            
            # Asegurarse de guardar los resultados
            self.verificar_y_guardar_resultados(folder_ciclo, ciclo_timestamp)
            
            # Cerrar la barra de progreso
            if mostrar_progreso and pbar:
                pbar.close()
                
            # Mostrar resumen
            if mostrar_progreso:
                print(f"\n✅ Proceso completado: {self.imagenes_procesadas}/{self.total_imagenes} imágenes procesadas")
                print(f"📊 Resultados guardados en: {os.path.join(RESULTADOS_FOLDER, f'{ciclo_timestamp}.json')}")

        except Exception as e:
            logging.critical(f"[{origen}] Error crítico en ejecutar_ronda: {e}")
            traceback.print_exc()
            if mostrar_progreso and pbar:
                pbar.close()
        finally:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)

    def limpiar_directorios_una_vez(self):
        try:
            for carpeta in [SAVE_FOLDER, RESULTADOS_FOLDER]:
                if os.path.exists(carpeta):
                    shutil.rmtree(carpeta)
                    logging.info(f"🧹 Carpeta eliminada: {carpeta}")
                os.makedirs(carpeta, exist_ok=True)
        except Exception as e:
            logging.error(f"❌ Error al limpiar directorios en modo una_vez: {e}")
            traceback.print_exc()

    def generar_y_subir_csv(self, json_path, csv_path, event_id):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

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

            logging.info(f"📄 CSV generado en: {csv_path}")

            if not SUBIR_SQL:
                logging.info("Subida a SQL desactivada (SUBIR_SQL = False)")
                return

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
    parser = argparse.ArgumentParser(description="Sistema de captura y clasificación de imágenes para Danone")
    parser.add_argument("--modo", choices=["una_vez", "periodico", "monitor"], default="una_vez", 
                       help="Modo de ejecución (una_vez = ejecución única, periodico = ejecución continua, monitor = modo visual)")
    parser.add_argument("--entorno", choices=["dev", "prod"], default="dev", help="Entorno de ejecución")
    parser.add_argument("--progreso", action="store_true", help="Mostrar barra de progreso")
    parser.add_argument("--config", type=str, help="Ruta a archivo de configuración personalizado")
    parser.add_argument("--camaras", type=str, help="IDs de cámaras separados por coma (ej: 0,1,2)")
    args = parser.parse_args()

    # Aplicar configuración según el entorno
    aplicar_configuracion_entorno(args.entorno)
    
    # Sobreescribir configuración con argumentos CLI si se proporcionan
    if args.camaras:
        try:
            CAMARAS_SELECCIONADAS = [int(cam) for cam in args.camaras.split(',')]
            logging.info(f"Cámaras seleccionadas manualmente: {CAMARAS_SELECCIONADAS}")
        except ValueError:
            logging.error("Formato incorrecto para --camaras. Debe ser lista de números separados por comas.")
    
    # Actualizar configuración de la base de datos
    DB_CONFIG['tabla'] = TABLA_SQL

    # Verificar si ya hay una ejecución en curso antes de iniciar
    if os.path.exists(LOCK_FILE):
        logging.warning(f"Ya hay una ejecución en curso (archivo {LOCK_FILE} detectado). Abortando.")
        sys.exit(1)

    capturador = CapturadorRafaga(CAMARAS_SELECCIONADAS)

    if args.modo == "una_vez":
        logging.info(f"Modo: UNA VEZ (Entorno: {args.entorno})")
        if args.progreso:
            print("="*50)
            print(f"🚀 INICIANDO CAPTURA DE IMÁGENES (Entorno: {args.entorno})")
            print("="*50)
        logging.info("Limpiando directorios...")
        capturador.limpiar_directorios_una_vez()
        capturador.ejecutar_ronda(origen="una_vez", mostrar_progreso=args.progreso)
        capturador.eliminar_fotos_antiguas()
        if args.progreso:
            print("="*50)
            print("✅ PROCESO TERMINADO")
            print("="*50)
        logging.info("Proceso terminado.")
    
    elif args.modo == "periodico":
        logging.info(f"Modo: PERIODICO (Entorno: {args.entorno})")
        if args.progreso:
            print("="*50)
            print(f"🔄 INICIANDO MODO PERIÓDICO (Entorno: {args.entorno})")
            print(f"Intervalo: {PHOTO_INTERVAL // 60} minutos")
            print("="*50)

        def loop_captura():
            while True:
                try:
                    start_time = time.time()
                    if args.progreso:
                        print(f"\n🔄 Iniciando captura: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    capturador.ejecutar_ronda(origen="periodico", mostrar_progreso=args.progreso)
                    capturador.eliminar_fotos_antiguas()
                    
                    # Calcular el tiempo que falta para la próxima ejecución
                    elapsed_time = time.time() - start_time
                    sleep_time = max(1, PHOTO_INTERVAL - elapsed_time)
                    
                    if args.progreso:
                        print(f"💤 Esperando {sleep_time:.1f} segundos para la próxima captura...")
                    
                    time.sleep(sleep_time)
                except KeyboardInterrupt:
                    logging.info("Proceso interrumpido por el usuario (Ctrl+C)")
                    if args.progreso:
                        print("\n✋ Proceso interrumpido por el usuario")
                    break
                except Exception as e:
                    logging.critical(f"Error en bucle principal: {str(e)}")
                    traceback.print_exc()
                    time.sleep(60)  # Esperar un minuto antes de reintentar
        
        # Iniciar el bucle en un hilo separado
        thread = threading.Thread(target=loop_captura)
        thread.daemon = True
        thread.start()
        
        try:
            # Mantener el hilo principal vivo
            while thread.is_alive():
                thread.join(1)
        except KeyboardInterrupt:
            logging.info("Proceso principal interrumpido por el usuario")
            if args.progreso:
                print("\n✋ Proceso principal interrumpido")
            
    elif args.modo == "monitor":
        logging.info(f"Modo: MONITOR (Entorno: {args.entorno})")
        print("="*50)
        print(f"👁️ INICIANDO MODO MONITOR (Entorno: {args.entorno})")
        print("="*50)
        
        try:
            # Verificar cámaras disponibles
            camaras_disponibles = capturador.verificar_camaras()
            if not camaras_disponibles:
                print("❌ No se encontraron cámaras disponibles")
                sys.exit(1)
                
            # Mostrar feed en vivo de las cámaras
            while True:
                for cam_id in camaras_disponibles:
                    try:
                        cap = cv2.VideoCapture(cam_id)
                        ret, frame = cap.read()
                        if ret:
                            if APLICAR_TRANSFORMACION:
                                frame = cv2.resize(frame, IMG_SIZE)
                                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            
                            cv2.imshow(f"Cámara {cam_id}", frame)
                        cap.release()
                    except Exception as e:
                        logging.error(f"Error al mostrar cámara {cam_id}: {e}")
                
                # Presionar 'q' para salir, 'c' para capturar
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    print("📸 Capturando imágenes...")
                    capturador.ejecutar_ronda(origen="monitor_manual", mostrar_progreso=True)
                    print("✅ Captura completa")
                    
            cv2.destroyAllWindows()
            
        except KeyboardInterrupt:
            logging.info("Modo monitor interrumpido por el usuario")
            print("\n✋ Modo monitor interrumpido")
            cv2.destroyAllWindows()
        except Exception as e:
            logging.critical(f"Error en modo monitor: {e}")
            traceback.print_exc()
            cv2.destroyAllWindows()
    
    else:
        logging.error(f"Modo desconocido: {args.modo}")
        print(f"❌ Modo no válido: {args.modo}")