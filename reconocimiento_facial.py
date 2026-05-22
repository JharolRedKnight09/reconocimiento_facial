import cv2
import numpy as np
import pickle
import os
import math
import time
import threading
from datetime import datetime
from insightface.app import FaceAnalysis

import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk

# ----- CONFIGURACION INICIAL -----
DB_PATH = "base_datos_caras.pkl"
DIR_INTRUSOS = "alertas_intrusion/"
UMBRAL_SIMILITUD = 0.85

os.makedirs(DIR_INTRUSOS, exist_ok=True)

# Configuracion Visual
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ----- INICIALIZACION DE MODELOS -----
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    import mediapipe as mp
    print(f"[DEBUG] Usando MediaPipe desde: {mp.__file__}")
except AttributeError:
    print("[ERROR] Modulo MediaPipe corrupto. Siga los pasos de reinstalacion sugeridos anteriormente.")
    exit(1)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print("Inicializando ArcFace (Modelo buffalo_l)...")
app_arcface = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_arcface.prepare(ctx_id=0, det_size=(640, 640))


# ----- GESTION DE BASE DE DATOS LOCAL -----
def cargar_bd():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def guardar_bd(bd):
    with open(DB_PATH, 'wb') as f:
        pickle.dump(bd, f)

def calcular_similitud_coseno(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    return dot_product / (norm_a * norm_b)


def validar_calidad_rostro(rostro_landmarks, frame_width, frame_height):
    nariz = rostro_landmarks.landmark[1]
    ojo_izq = rostro_landmarks.landmark[33]
    ojo_der = rostro_landmarks.landmark[263]
    x_nariz = nariz.x * frame_width
    x_ojo_izq = ojo_izq.x * frame_width
    x_ojo_der = ojo_der.x * frame_width
    dist_izq = abs(x_nariz - x_ojo_izq)
    dist_der = abs(x_ojo_der - x_nariz)
    if dist_izq == 0 or dist_der == 0:
        return False
    ratio = dist_izq / dist_der if dist_izq < dist_der else dist_der / dist_izq
    if ratio < 0.65:
        return False
    return True


# ----- CLASE PRINCIPAL UI -----
class BiometricSystemApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sistema Biometrico Facial Local")
        self.geometry("1100x700")
        self.protocol("WM_DELETE_WINDOW", self.on_cerrar)

        # Estados logicos
        self.cap = None
        self.fase_actual = None # "ENROLAR" o "RECONOCER"
        self.nombre_enrolar = ""
        self.frame_count = 0
        self.faces_cacheadas = []
        self.frames_acceso_concedido = 0
        self.usuario_concedido = ""
        self.ultimo_guardado = 0
        self.calidad_ok = False
        self.angulo_alineacion = 0.0
        self.bd_en_memoria = cargar_bd()
        self.tiempo_ultimo_acceso = 0
        self.procesando_arcface = False
        self._camera_lock = threading.Lock()

        self.setup_ui()

    def setup_ui(self):
        # Grid layout (1 fila, 2 columnas)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3) # Panel de video
        self.grid_columnconfigure(1, weight=1) # Panel de menu

        # Panel Izquierdo: Video y Logs
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.lbl_video = ctk.CTkLabel(self.video_frame, text="Sistema en Reposo\nPulse una opcion a la derecha", font=("Roboto", 20))
        self.lbl_video.grid(row=0, column=0, sticky="nsew", pady=10)

        self.lbl_estado = ctk.CTkLabel(self.video_frame, text="OK - Base de datos cargada correctamente.", font=("Roboto", 14), text_color="green")
        self.lbl_estado.grid(row=1, column=0, pady=10)

        # Panel Derecho: Botones y Control
        self.menu_frame = ctk.CTkFrame(self)
        self.menu_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)

        lbl_titulo = ctk.CTkLabel(self.menu_frame, text="PANEL DE CONTROL", font=("Roboto", 24, "bold"))
        lbl_titulo.pack(pady=30)

        self.btn_acceso = ctk.CTkButton(self.menu_frame, text="Iniciar Acceso Automatico", font=("Roboto", 16, "bold"), height=50, command=self.comenzar_reconocimiento)
        self.btn_acceso.pack(fill="x", padx=20, pady=10)

        self.btn_iniciar_enrolar = ctk.CTkButton(self.menu_frame, text="Enrolar Usuario", font=("Roboto", 16), height=50, fg_color="#333", hover_color="#555", command=self.comenzar_enrolamiento)
        self.btn_iniciar_enrolar.pack(fill="x", padx=20, pady=10)
        
        self.btn_capturar = ctk.CTkButton(self.menu_frame, text="Capturar Rostro", font=("Roboto", 16, "bold"), height=50, fg_color="green", hover_color="#006400", state="disabled", command=self.ejecutar_captura_enrolamiento)
        self.btn_capturar.pack(fill="x", padx=20, pady=10)

        self.btn_eliminar = ctk.CTkButton(self.menu_frame, text="Eliminar Acceso", font=("Roboto", 16), height=50, fg_color="#333", hover_color="#8b0000", command=self.eliminar_usuario_ui)
        self.btn_eliminar.pack(fill="x", padx=20, pady=10)

        self.btn_detener = ctk.CTkButton(self.menu_frame, text="Detener Camara", font=("Roboto", 16), height=50, fg_color="red", hover_color="#8b0000", state="disabled", command=self.detener_camara)
        self.btn_detener.pack(fill="x", padx=20, pady=10)

    def log(self, mensaje, color="white"):
        self.lbl_estado.configure(text=mensaje, text_color=color)

    def detener_camara(self):
        self.fase_actual = None # Detiene el bucle de video
        
        with self._camera_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        
        self.btn_detener.configure(state="disabled")
        self.btn_capturar.configure(state="disabled")
        self.btn_acceso.configure(state="normal")
        self.btn_iniciar_enrolar.configure(state="normal")
        # Forzar la limpieza visual
        self.lbl_video.configure(image="", text="Camara Detenida")
        self.log("Sistema detenido.", "gray")

    def iniciar_camara(self):
        with self._camera_lock:
            if self.cap is not None:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error de Hardware", "No se puede acceder a la camara web.")
                self.cap = None
                return False
        
        self.btn_detener.configure(state="normal")
        self.btn_acceso.configure(state="disabled")
        self.btn_iniciar_enrolar.configure(state="disabled")
        self.frame_count = 0
        return True

    def comenzar_enrolamiento(self):
        dialog = ctk.CTkInputDialog(text="Ingrese el nombre completo del usuario:", title="Registro")
        nombre = dialog.get_input()
        
        if not nombre or not nombre.strip():
            return
            
        self.nombre_enrolar = nombre.strip()
        self.fase_actual = "ENROLAR"
        self.btn_capturar.configure(state="normal")
        
        if self.iniciar_camara():
            self.log(f"Mire a la camara y pulse CAPTURAR.\nRegistrando a: {self.nombre_enrolar}", "yellow")
            self.loop_video()

    def ejecutar_captura_enrolamiento(self):
        if self.fase_actual == "ENROLAR" and self.calidad_ok:
            self.log("Procesando...", "yellow")
            
            with self._camera_lock:
                if self.cap is None or not self.cap.isOpened():
                    return
                ret, frame = self.cap.read()
            
            if not ret: return
            
            h, w, _ = frame.shape
            centro_rostro = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(centro_rostro, self.angulo_alineacion, 1.0)
            frame_alineado = cv2.warpAffine(frame, M, (w, h))

            # Extraccion del modelo 512D
            faces = app_arcface.get(frame_alineado)
            
            if len(faces) == 0:
                self.log("Error: ArcFace no hallo rostro al capturar. Repita.", "red")
                return
                
            embedding_512 = faces[0].embedding
            
            # Guardamos a nivel local
            self.bd_en_memoria[self.nombre_enrolar] = embedding_512
            guardar_bd(self.bd_en_memoria)
            
            messagebox.showinfo("Exito", f"Usuario '{self.nombre_enrolar}' enrolado satisfactoriamente.")
            self.detener_camara()
        else:
            self.log("Rostro no listo. Mire de frente antes de capturar.", "red")

    def comenzar_reconocimiento(self):
        if not self.bd_en_memoria:
            messagebox.showwarning("BD Vacia", "No hay usuarios registrados en el sistema.")
            return
            
        self.fase_actual = "RECONOCER"
        self.frames_acceso_concedido = 0
        self.usuario_concedido = ""
        
        if self.iniciar_camara():
            self.log("Mire a la camara para validar su identidad y abrir la puerta.", "orange")
            # En base a la peticion del usuario, no detener automaticamente la camara para escanear constantemente
            self.loop_video()

    def eliminar_usuario_ui(self):
        if not self.bd_en_memoria:
            messagebox.showwarning("Aviso", "No hay usuarios registrados en el sistema.")
            return
            
        nombres = "\n".join(f"- {name}" for name in self.bd_en_memoria.keys())
        dialog = ctk.CTkInputDialog(text=f"Usuarios actuales:\n{nombres}\n\nEscriba EXACTAMENTE el nombre a eliminar:", title="Eliminar Usuario")
        nombre = dialog.get_input()
        
        if nombre:
            if nombre in self.bd_en_memoria:
                del self.bd_en_memoria[nombre]
                guardar_bd(self.bd_en_memoria)
                messagebox.showinfo("Eliminado", f"El usuario {nombre} fue eliminado del sistema.")
            else:
                messagebox.showerror("Error", f"Usuario '{nombre}' no encontrado.")

    def procesar_arcface_en_hilo(self, frame_busqueda):
        try:
            self.faces_cacheadas = app_arcface.get(frame_busqueda)
        except Exception as e:
            pass
        finally:
            self.procesando_arcface = False

    def loop_video(self):
        # Proteccion basica
        if self.fase_actual is None:
            return
            
        with self._camera_lock:
            if self.cap is None or not self.cap.isOpened():
                return
            ret, frame = self.cap.read()
            
        if not ret:
            self.after(10, self.loop_video)
            return

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar postura con MediaPipe
        results = face_mesh.process(frame_rgb)
        
        self.calidad_ok = False
        self.angulo_alineacion = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            self.calidad_ok = validar_calidad_rostro(landmarks, w, h)
            
            ojo_izq = landmarks.landmark[33]
            ojo_der = landmarks.landmark[263]
            dy = (ojo_der.y - ojo_izq.y) * h
            dx = (ojo_der.x - ojo_izq.x) * w
            self.angulo_alineacion = math.degrees(math.atan2(dy, dx))

            if self.fase_actual == "ENROLAR":
                # Guias visuales para enrolamiento
                color_line = (0, 255, 0) if self.calidad_ok else (255, 0, 0)
                cv2.line(frame, (int(ojo_izq.x * w), int(ojo_izq.y * h)), 
                         (int(ojo_der.x * w), int(ojo_der.y * h)), color_line, 2)
                
                estado = "Rostro Valido: PUEDE CAPTURAR" if self.calidad_ok else "Rostro Ladeado: MIRE AL FRENTE"
                cv2.putText(frame, estado, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_line, 2)
                
            elif self.fase_actual == "RECONOCER":
                if not self.calidad_ok:
                    self.frames_acceso_concedido = 0
                    cv2.putText(frame, "POSTURA NO VALIDA. Mire de frente.", (20, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    # Optimizacion Multi-Threading: ArcFace no bloquea la UI principal
                    if not self.procesando_arcface:
                        self.procesando_arcface = True
                        frame_busqueda = frame.copy()
                        threading.Thread(target=self.procesar_arcface_en_hilo, args=(frame_busqueda,), daemon=True).start()
                        
                    for face in self.faces_cacheadas:
                        bbox = face.bbox.astype(int)
                        emb_capturado = face.embedding
                        mejor_similitud = -1.0
                        mejor_match = "Desconocido"

                        # Comparar el rostro extraido con los registros locales en memoria
                        for nombre_bd, emb_bd in self.bd_en_memoria.items():
                            similitud = calcular_similitud_coseno(emb_capturado, emb_bd)
                            if similitud > mejor_similitud:
                                mejor_similitud = similitud
                                mejor_match = nombre_bd
                                
                        if mejor_similitud >= UMBRAL_SIMILITUD:
                            # Acceso Aprobado Visible
                            color = (0, 255, 0)
                            texto = f"{mejor_match} - {(mejor_similitud*100):.1f}%"
                            self.frames_acceso_concedido += 1
                            self.usuario_concedido = mejor_match
                        else:
                            # Acceso Denegado
                            color = (0, 0, 255)
                            texto = "Denegado"
                            self.frames_acceso_concedido = 0
                            
                            t_actual = time.time()
                            if t_actual - self.ultimo_guardado > 3:
                                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                path_foto = os.path.join(DIR_INTRUSOS, f"intruso_{stamp}.jpg")
                                cv2.imwrite(path_foto, frame)
                                self.log(f"Intruso guardado: {path_foto}", "red")
                                self.ultimo_guardado = t_actual

                        # Renderizado del recuadro
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[0]+250, bbox[1]), color, -1)
                        cv2.putText(frame, texto, (bbox[0]+5, bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                    # Mecanica de desbloqueo Continuo sin detener la camara
                    if self.frames_acceso_concedido >= 3:
                        t_actual = time.time()
                        if t_actual - self.tiempo_ultimo_acceso > 5:
                            self.log(f"ACCESO CONCEDIDO A: {self.usuario_concedido}. Puerta abierta.", "#00ff00")
                            self.tiempo_ultimo_acceso = t_actual
                        
                        # --- MEJORA: Banner gigante en pantalla indicando BIENVENIDO ---
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, h//2 - 70), (w, h//2 + 70), (0, 200, 0), -1)
                        # Aplicar opacidad al banner verde (60%)
                        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                        
                        txt_bienvenida = "BIENVENIDO AL SISTEMA"
                        txt_usuario = self.usuario_concedido.upper()
                        
                        # Centrar texto 1
                        tam1, _ = cv2.getTextSize(txt_bienvenida, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        cv2.putText(frame, txt_bienvenida, ((w - tam1[0]) // 2, h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                        
                        # Centrar texto 2
                        tam2, _ = cv2.getTextSize(txt_usuario, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                        cv2.putText(frame, txt_usuario, ((w - tam2[0]) // 2, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

        else:
            self.frames_acceso_concedido = 0

        self.frame_count += 1
        self.mostrar_frame_en_ui(frame)

        # Si seguimos en un modo activo, pedimos el siguiente frame en ~15ms
        if self.fase_actual is not None:
            self.after(15, self.loop_video)

    def mostrar_frame_en_ui(self, frame_cv2):
        # Convertir BGR de OpenCV a RGB para Pillow
        rgb_image = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Redimensionamiento seguro manteniendo relacion de aspecto
        w_max, h_max = 800, 600
        pil_image.thumbnail((w_max, h_max))
        
        # Usamos PIL ImageTk compatibilidad pura
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(pil_image.width, pil_image.height))
        self.lbl_video.configure(image=ctk_image, text="")

    def on_cerrar(self):
        if self.cap is not None:
            self.cap.release()
        self.destroy()

# ----- EJECUCION PRINCIPAL -----
if __name__ == "__main__":
    app = BiometricSystemApp()
    app.mainloop()