import os
import configparser
import cv2
import customtkinter as ctk
import numpy as np

from datetime import datetime
from PIL import Image
from skimage.metrics import structural_similarity


class CamaraVisionMaster:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.ventana = ctk.CTk()
        self.ventana.title("Cámara de visión - Comparación MASTER")
        self.ventana.geometry("1280x820")
        self.ventana.minsize(1100, 700)

        self.color_fondo = "#21233C"
        self.color_panel = "#292C47"
        self.color_borde = "#454B70"

        self.ventana.configure(fg_color=self.color_fondo)

        self.camara = None
        self.frame_actual = None
        self.imagen_master = None

        self.comparacion_activa = False
        self.actualizacion_programada = None

        # Umbral inicial de aceptación.
        # Puede ajustarse según las pruebas reales.
        self.umbral_similitud = 90.0

        # Región de inspección.
        self.roi_x1 = 223
        self.roi_y1 = 110
        self.roi_x2 = 429
        self.roi_y2 = 138

        # Variables para configurar el ROI directamente sobre la imagen.
        self.configurando_roi = False
        self.roi_arrastrando = False
        self.roi_inicio = None
        self.roi_temporal = None

        # Datos usados para convertir las coordenadas de pantalla
        # a coordenadas reales de la cámara.
        self.escala_video = 1.0
        self.offset_video_x = 0
        self.offset_video_y = 0
        self.ancho_video_mostrado = 640
        self.alto_video_mostrado = 480

        self.carpeta_master = "master"
        self.carpeta_resultados = "resultados"

        os.makedirs(self.carpeta_master, exist_ok=True)
        os.makedirs(self.carpeta_resultados, exist_ok=True)

        self.ruta_master = os.path.join(
            self.carpeta_master,
            "pieza_master.jpg"
        )

        self.ruta_configuracion = "config.ini"
        self.cargar_configuracion_roi()

        self.crear_interfaz()
        self.cargar_master_existente()
        self.iniciar_camara()

        self.ventana.protocol(
            "WM_DELETE_WINDOW",
            self.salir
        )

    def crear_interfaz(self):
        self.ventana.grid_columnconfigure(0, weight=1)
        self.ventana.grid_rowconfigure(1, weight=1)

        # Encabezado.
        frame_encabezado = ctk.CTkFrame(
            self.ventana,
            height=75,
            fg_color=self.color_panel,
            corner_radius=0
        )
        frame_encabezado.grid(
            row=0,
            column=0,
            sticky="ew"
        )

        frame_encabezado.grid_columnconfigure(0, weight=1)

        titulo = ctk.CTkLabel(
            frame_encabezado,
            text="CÁMARA DE VISIÓN",
            font=("Arial", 28, "bold"),
            text_color="white"
        )
        titulo.grid(
            row=0,
            column=0,
            padx=25,
            pady=(12, 0),
            sticky="w"
        )

        subtitulo = ctk.CTkLabel(
            frame_encabezado,
            text="Comparación automática contra pieza MASTER",
            font=("Arial", 15),
            text_color="#AEB4D0"
        )
        subtitulo.grid(
            row=1,
            column=0,
            padx=25,
            pady=(0, 10),
            sticky="w"
        )

        # Contenido.
        frame_contenido = ctk.CTkFrame(
            self.ventana,
            fg_color="transparent"
        )
        frame_contenido.grid(
            row=1,
            column=0,
            padx=20,
            pady=20,
            sticky="nsew"
        )

        frame_contenido.grid_columnconfigure(0, weight=4)
        frame_contenido.grid_columnconfigure(1, weight=1)
        frame_contenido.grid_rowconfigure(0, weight=1)

        # Panel de cámara.
        self.frame_camara = ctk.CTkFrame(
            frame_contenido,
            fg_color="#11131F",
            border_width=2,
            border_color=self.color_borde,
            corner_radius=12
        )
        self.frame_camara.grid(
            row=0,
            column=0,
            padx=(0, 15),
            sticky="nsew"
        )

        self.frame_camara.grid_columnconfigure(0, weight=1)
        self.frame_camara.grid_rowconfigure(0, weight=1)

        self.label_video = ctk.CTkLabel(
            self.frame_camara,
            text="Iniciando cámara...",
            font=("Arial", 22),
            text_color="#AEB4D0"
        )
        self.label_video.grid(
            row=0,
            column=0,
            padx=10,
            pady=10,
            sticky="nsew"
        )

        # Eventos del mouse para dibujar el ROI sobre la imagen.
        self.label_video.bind(
            "<ButtonPress-1>",
            self.iniciar_dibujo_roi
        )
        self.label_video.bind(
            "<B1-Motion>",
            self.actualizar_dibujo_roi
        )
        self.label_video.bind(
            "<ButtonRelease-1>",
            self.finalizar_dibujo_roi
        )

        # Panel derecho.
        frame_control = ctk.CTkFrame(
            frame_contenido,
            width=270,
            fg_color=self.color_panel,
            border_width=1,
            border_color=self.color_borde,
            corner_radius=12
        )
        frame_control.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        frame_control.grid_propagate(False)
        frame_control.grid_columnconfigure(0, weight=1)

        label_estado_titulo = ctk.CTkLabel(
            frame_control,
            text="RESULTADO",
            font=("Arial", 18, "bold"),
            text_color="#AEB4D0"
        )
        label_estado_titulo.grid(
            row=0,
            column=0,
            padx=20,
            pady=(25, 5)
        )

        self.label_resultado = ctk.CTkLabel(
            frame_control,
            text="SIN MASTER",
            width=210,
            height=75,
            corner_radius=10,
            fg_color="#555555",
            font=("Arial", 25, "bold"),
            text_color="white"
        )
        self.label_resultado.grid(
            row=1,
            column=0,
            padx=20,
            pady=10
        )

        self.label_similitud = ctk.CTkLabel(
            frame_control,
            text="Similitud: --- %",
            font=("Arial", 21, "bold"),
            text_color="white"
        )
        self.label_similitud.grid(
            row=2,
            column=0,
            padx=20,
            pady=(5, 20)
        )

        self.label_master = ctk.CTkLabel(
            frame_control,
            text="MASTER: No cargada",
            font=("Arial", 14),
            text_color="#FFB86C",
            wraplength=220
        )
        self.label_master.grid(
            row=3,
            column=0,
            padx=20,
            pady=10
        )

        label_umbral = ctk.CTkLabel(
            frame_control,
            text="Umbral de aceptación",
            font=("Arial", 15, "bold"),
            text_color="#AEB4D0"
        )
        label_umbral.grid(
            row=4,
            column=0,
            padx=20,
            pady=(20, 5)
        )

        self.label_valor_umbral = ctk.CTkLabel(
            frame_control,
            text=f"{self.umbral_similitud:.0f} %",
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        self.label_valor_umbral.grid(
            row=5,
            column=0,
            padx=20
        )

        self.slider_umbral = ctk.CTkSlider(
            frame_control,
            from_=50,
            to=100,
            number_of_steps=50,
            command=self.cambiar_umbral
        )
        self.slider_umbral.set(self.umbral_similitud)
        self.slider_umbral.grid(
            row=6,
            column=0,
            padx=25,
            pady=(5, 25),
            sticky="ew"
        )

        self.boton_configurar_roi = ctk.CTkButton(
            frame_control,
            text="CONFIGURAR ÁREA",
            height=45,
            font=("Arial", 15, "bold"),
            fg_color="#6C5CE7",
            hover_color="#5546C7",
            command=self.alternar_configuracion_roi
        )
        self.boton_configurar_roi.grid(
            row=7,
            column=0,
            padx=20,
            pady=7,
            sticky="ew"
        )

        self.boton_master = ctk.CTkButton(
            frame_control,
            text="CAPTURAR MASTER",
            height=45,
            font=("Arial", 15, "bold"),
            command=self.capturar_master
        )
        self.boton_master.grid(
            row=8,
            column=0,
            padx=20,
            pady=7,
            sticky="ew"
        )

        self.boton_comparar = ctk.CTkButton(
            frame_control,
            text="INICIAR COMPARACIÓN",
            height=45,
            font=("Arial", 15, "bold"),
            fg_color="#218C4A",
            hover_color="#176A38",
            command=self.alternar_comparacion
        )
        self.boton_comparar.grid(
            row=9,
            column=0,
            padx=20,
            pady=7,
            sticky="ew"
        )

        self.boton_guardar = ctk.CTkButton(
            frame_control,
            text="GUARDAR RESULTADO",
            height=45,
            font=("Arial", 15, "bold"),
            fg_color="#8A6D1D",
            hover_color="#6C5515",
            command=self.guardar_resultado
        )
        self.boton_guardar.grid(
            row=10,
            column=0,
            padx=20,
            pady=7,
            sticky="ew"
        )

        self.boton_salir = ctk.CTkButton(
            frame_control,
            text="SALIR",
            height=45,
            font=("Arial", 15, "bold"),
            fg_color="#A93232",
            hover_color="#7F2626",
            command=self.salir
        )
        self.boton_salir.grid(
            row=11,
            column=0,
            padx=20,
            pady=(7, 20),
            sticky="ew"
        )

    def cargar_configuracion_roi(self):
        """
        Carga las coordenadas del ROI guardadas en config.ini.
        Si el archivo no existe o contiene datos inválidos,
        conserva los valores definidos en el código.
        """
        if not os.path.exists(self.ruta_configuracion):
            return

        config = configparser.ConfigParser()

        try:
            config.read(
                self.ruta_configuracion,
                encoding="utf-8"
            )

            if "ROI" not in config:
                return

            x1 = config.getint("ROI", "x1")
            y1 = config.getint("ROI", "y1")
            x2 = config.getint("ROI", "x2")
            y2 = config.getint("ROI", "y2")

            if x2 <= x1 or y2 <= y1:
                return

            self.roi_x1 = x1
            self.roi_y1 = y1
            self.roi_x2 = x2
            self.roi_y2 = y2

        except (
            configparser.Error,
            ValueError,
            OSError
        ):
            pass

    def guardar_configuracion_roi(self):
        """
        Guarda el ROI para recuperarlo al abrir nuevamente
        la aplicación.
        """
        config = configparser.ConfigParser()

        config["ROI"] = {
            "x1": str(self.roi_x1),
            "y1": str(self.roi_y1),
            "x2": str(self.roi_x2),
            "y2": str(self.roi_y2)
        }

        try:
            with open(
                self.ruta_configuracion,
                "w",
                encoding="utf-8"
            ) as archivo:
                config.write(archivo)

            return True

        except OSError:
            return False

    def alternar_configuracion_roi(self):
        """
        Activa o cancela el modo para dibujar el área de
        inspección directamente sobre la imagen.
        """
        self.configurando_roi = not self.configurando_roi
        self.roi_arrastrando = False
        self.roi_inicio = None
        self.roi_temporal = None

        if self.configurando_roi:
            self.comparacion_activa = False

            self.boton_comparar.configure(
                text="INICIAR COMPARACIÓN",
                fg_color="#218C4A",
                hover_color="#176A38"
            )

            self.boton_configurar_roi.configure(
                text="CANCELAR AJUSTE",
                fg_color="#B3261E",
                hover_color="#871B16"
            )

            self.label_video.configure(
                cursor="crosshair"
            )

            self.label_resultado.configure(
                text="AJUSTAR ROI",
                fg_color="#6C5CE7"
            )

            self.label_similitud.configure(
                text="Arrastra sobre la imagen"
            )

        else:
            self.boton_configurar_roi.configure(
                text="CONFIGURAR ÁREA",
                fg_color="#6C5CE7",
                hover_color="#5546C7"
            )

            self.label_video.configure(
                cursor=""
            )

            self.label_resultado.configure(
                text="CANCELADO",
                fg_color="#555555"
            )

            self.label_similitud.configure(
                text="Similitud: --- %"
            )

    def convertir_pantalla_a_frame(self, x_pantalla, y_pantalla):
        """
        Convierte una posición del mouse sobre el CTkLabel
        a coordenadas reales del frame de la cámara.
        """
        if self.frame_actual is None:
            return None

        x_relativa = x_pantalla - self.offset_video_x
        y_relativa = y_pantalla - self.offset_video_y

        if (
            x_relativa < 0
            or y_relativa < 0
            or x_relativa >= self.ancho_video_mostrado
            or y_relativa >= self.alto_video_mostrado
        ):
            return None

        alto_frame, ancho_frame = self.frame_actual.shape[:2]

        x_frame = int(
            x_relativa / self.escala_video
        )
        y_frame = int(
            y_relativa / self.escala_video
        )

        x_frame = max(
            0,
            min(x_frame, ancho_frame - 1)
        )
        y_frame = max(
            0,
            min(y_frame, alto_frame - 1)
        )

        return x_frame, y_frame

    def iniciar_dibujo_roi(self, event):
        if not self.configurando_roi:
            return

        punto = self.convertir_pantalla_a_frame(
            event.x,
            event.y
        )

        if punto is None:
            return

        self.roi_arrastrando = True
        self.roi_inicio = punto
        self.roi_temporal = (
            punto[0],
            punto[1],
            punto[0],
            punto[1]
        )

    def actualizar_dibujo_roi(self, event):
        if (
            not self.configurando_roi
            or not self.roi_arrastrando
            or self.roi_inicio is None
        ):
            return

        punto = self.convertir_pantalla_a_frame(
            event.x,
            event.y
        )

        if punto is None:
            return

        x_inicio, y_inicio = self.roi_inicio
        x_actual, y_actual = punto

        self.roi_temporal = (
            min(x_inicio, x_actual),
            min(y_inicio, y_actual),
            max(x_inicio, x_actual),
            max(y_inicio, y_actual)
        )

    def finalizar_dibujo_roi(self, event):
        if (
            not self.configurando_roi
            or not self.roi_arrastrando
            or self.roi_inicio is None
        ):
            return

        punto = self.convertir_pantalla_a_frame(
            event.x,
            event.y
        )

        self.roi_arrastrando = False

        if punto is None:
            self.roi_temporal = None
            self.roi_inicio = None
            return

        x_inicio, y_inicio = self.roi_inicio
        x_final, y_final = punto

        x1 = min(x_inicio, x_final)
        y1 = min(y_inicio, y_final)
        x2 = max(x_inicio, x_final)
        y2 = max(y_inicio, y_final)

        # Evita guardar áreas demasiado pequeñas creadas por
        # un clic accidental.
        ancho_roi = x2 - x1
        alto_roi = y2 - y1

        if ancho_roi < 20 or alto_roi < 20:
            self.roi_temporal = None
            self.roi_inicio = None

            self.label_resultado.configure(
                text="ROI PEQUEÑO",
                fg_color="#A66A00"
            )

            self.label_similitud.configure(
                text="Dibuja un área mayor"
            )
            return

        self.roi_x1 = x1
        self.roi_y1 = y1
        self.roi_x2 = x2
        self.roi_y2 = y2

        self.roi_temporal = None
        self.roi_inicio = None
        self.configurando_roi = False

        guardado = self.guardar_configuracion_roi()

        # Una MASTER anterior deja de ser válida al cambiar el ROI.
        self.imagen_master = None

        try:
            if os.path.exists(self.ruta_master):
                os.remove(self.ruta_master)
        except OSError:
            pass

        self.boton_configurar_roi.configure(
            text="CONFIGURAR ÁREA",
            fg_color="#6C5CE7",
            hover_color="#5546C7"
        )

        self.label_video.configure(
            cursor=""
        )

        self.label_resultado.configure(
            text="ROI GUARDADO",
            fg_color="#315A77"
        )

        if guardado:
            self.label_similitud.configure(
                text=f"ROI: {ancho_roi} x {alto_roi} px"
            )
        else:
            self.label_similitud.configure(
                text="ROI aplicado; no se guardó config.ini"
            )

        self.label_master.configure(
            text="MASTER: Capture una nueva",
            text_color="#FFB86C"
        )

    def iniciar_camara(self):
        self.camara = cv2.VideoCapture(
            1,
            cv2.CAP_DSHOW
        )

        if not self.camara.isOpened():
            self.label_video.configure(
                text="No fue posible abrir la cámara."
            )
            return

        self.camara.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )
        self.camara.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )
        self.camara.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.actualizar_video()

    def cargar_master_existente(self):
        if not os.path.exists(self.ruta_master):
            return

        imagen = cv2.imread(self.ruta_master)

        if imagen is None:
            return

        self.imagen_master = imagen

        self.label_master.configure(
            text="MASTER: Cargada",
            text_color="#6FE3A1"
        )

        self.label_resultado.configure(
            text="LISTO",
            fg_color="#315A77"
        )

    def cambiar_umbral(self, valor):
        self.umbral_similitud = float(valor)

        self.label_valor_umbral.configure(
            text=f"{self.umbral_similitud:.0f} %"
        )

    def obtener_roi(self, frame):
        alto, ancho = frame.shape[:2]

        x1 = max(0, min(self.roi_x1, ancho - 1))
        y1 = max(0, min(self.roi_y1, alto - 1))
        x2 = max(x1 + 1, min(self.roi_x2, ancho))
        y2 = max(y1 + 1, min(self.roi_y2, alto))

        return frame[y1:y2, x1:x2]

    def preparar_imagen(self, imagen):
        gris = cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2GRAY
        )

        gris = cv2.GaussianBlur(
            gris,
            (9, 9),
            0
        )

        return gris

    def comparar_con_master(self, roi_actual):
        if self.imagen_master is None:
            return None, None, 0, 0, 0

        master = self.imagen_master.copy()

        roi_actual = cv2.resize(
            roi_actual,
            (
                master.shape[1],
                master.shape[0]
            )
        )

        gris_master = self.preparar_imagen(master)
        gris_actual = self.preparar_imagen(roi_actual)

        similitud, mapa_similitud = structural_similarity(
            gris_master,
            gris_actual,
            full=True,
            data_range=255
        )

        porcentaje_similitud = similitud * 100

        # SSIM entrega 255 donde ambas imágenes son similares.
        mapa_similitud = (
            mapa_similitud * 255
        ).astype("uint8")

        # Invertimos para que las diferencias sean blancas.
        mapa_diferencias = cv2.bitwise_not(
            mapa_similitud
        )

        # Ajusta este valor para aumentar o disminuir sensibilidad.
        _, mascara = cv2.threshold(
            mapa_diferencias,
            90,
            255,
            cv2.THRESH_BINARY
        )

        # Eliminar ruido pequeño.
        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mascara = cv2.morphologyEx(
            mascara,
            cv2.MORPH_OPEN,
            kernel
        )

        mascara = cv2.morphologyEx(
            mascara,
            cv2.MORPH_CLOSE,
            kernel
        )

        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        imagen_resultado = roi_actual.copy()

        area_total_diferente = 0
        area_defecto_mayor = 0
        cantidad_defectos = 0

        # Ignora diferencias menores a este número de píxeles.
        area_minima_contorno = 30

        for contorno in contornos:
            area = cv2.contourArea(contorno)

            if area < area_minima_contorno:
                continue

            cantidad_defectos += 1
            area_total_diferente += area
            area_defecto_mayor = max(
                area_defecto_mayor,
                area
            )

            x, y, ancho, alto = cv2.boundingRect(
                contorno
            )

            cv2.rectangle(
                imagen_resultado,
                (x, y),
                (x + ancho, y + alto),
                (0, 0, 255),
                2
            )

        area_roi = (
            master.shape[0]
            * master.shape[1]
        )

        porcentaje_area_diferente = (
            area_total_diferente / area_roi
        ) * 100

        return (
            porcentaje_similitud,
            imagen_resultado,
            porcentaje_area_diferente,
            area_defecto_mayor,
            cantidad_defectos
        )

    def actualizar_video(self):
        if self.camara is None:
            return

        correcto, frame = self.camara.read()

        if not correcto or frame is None:
            self.actualizacion_programada = self.ventana.after(
                30,
                self.actualizar_video
            )
            return

        self.frame_actual = frame.copy()

        frame_mostrar = frame.copy()

        if (
            self.configurando_roi
            and self.roi_temporal is not None
        ):
            roi_dibujado = self.roi_temporal
            color_roi = (255, 0, 255)
            texto_roi = "NUEVA AREA"
        else:
            roi_dibujado = (
                self.roi_x1,
                self.roi_y1,
                self.roi_x2,
                self.roi_y2
            )
            color_roi = (0, 255, 255)
            texto_roi = "AREA DE INSPECCION"

        rx1, ry1, rx2, ry2 = roi_dibujado

        cv2.rectangle(
            frame_mostrar,
            (rx1, ry1),
            (rx2, ry2),
            color_roi,
            3
        )

        texto_y = max(
            25,
            ry1 - 15
        )

        cv2.putText(
            frame_mostrar,
            texto_roi,
            (rx1, texto_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_roi,
            2
        )

        if (
            self.comparacion_activa
            and self.imagen_master is not None
        ):
            roi_actual = self.obtener_roi(frame)

            (
                porcentaje,
                roi_resultado,
                porcentaje_diferente,
                area_defecto_mayor,
                cantidad_defectos
            ) = self.comparar_con_master(
                roi_actual
            )

            if porcentaje is not None:
                similitud_correcta = (
                    porcentaje >= self.umbral_similitud
                )

                # Reprobar cuando una sola diferencia sea suficientemente grande.
                sin_defecto_grande = (
                    area_defecto_mayor < 35
                )

                # Reprobar cuando el área total diferente sea relevante.
                area_diferente_aceptable = (
                    porcentaje_diferente < 0.08
                )

                aprobado = (
                    similitud_correcta
                    and sin_defecto_grande
                    and area_diferente_aceptable
                )

                if aprobado:
                    texto = "PASS"
                    color_bgr = (0, 255, 0)
                    color_interfaz = "#218C4A"
                else:
                    texto = "FAIL"
                    color_bgr = (0, 0, 255)
                    color_interfaz = "#B3261E"

                self.label_resultado.configure(
                    text=texto,
                    fg_color=color_interfaz
                )

                self.label_similitud.configure(
                    text=f"Similitud: {porcentaje:.2f} %"
                )

                roi_resultado = cv2.resize(
                    roi_resultado,
                    (
                        self.roi_x2 - self.roi_x1,
                        self.roi_y2 - self.roi_y1
                    )
                )

                frame_mostrar[
                    self.roi_y1:self.roi_y2,
                    self.roi_x1:self.roi_x2
                ] = roi_resultado

                cv2.rectangle(
                    frame_mostrar,
                    (self.roi_x1, self.roi_y1),
                    (self.roi_x2, self.roi_y2),
                    color_bgr,
                    4
                )

                cv2.putText(
                    frame_mostrar,
                    f"Diferencia: {porcentaje_diferente:.3f}%",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color_bgr,
                    2
                )

                cv2.putText(
                    frame_mostrar,
                    f"Defectos: {cantidad_defectos}",
                    (30, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color_bgr,
                    2
                )

        self.mostrar_frame(frame_mostrar)

        self.actualizacion_programada = self.ventana.after(
            20,
            self.actualizar_video
        )

    def mostrar_frame(self, frame):
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        imagen_pil = Image.fromarray(
            frame_rgb
        )

        ancho_disponible = max(
            self.frame_camara.winfo_width() - 20,
            640
        )

        alto_disponible = max(
            self.frame_camara.winfo_height() - 20,
            480
        )

        ancho_original, alto_original = imagen_pil.size

        escala = min(
            ancho_disponible / ancho_original,
            alto_disponible / alto_original
        )

        nuevo_ancho = max(
            1,
            int(ancho_original * escala)
        )

        nuevo_alto = max(
            1,
            int(alto_original * escala)
        )

        imagen_pil = imagen_pil.resize(
            (nuevo_ancho, nuevo_alto),
            Image.Resampling.LANCZOS
        )

        # Guardar la transformación usada para mostrar la imagen.
        # Es necesaria para interpretar correctamente el mouse.
        self.escala_video = escala
        self.ancho_video_mostrado = nuevo_ancho
        self.alto_video_mostrado = nuevo_alto

        ancho_label = max(
            self.label_video.winfo_width(),
            nuevo_ancho
        )
        alto_label = max(
            self.label_video.winfo_height(),
            nuevo_alto
        )

        self.offset_video_x = max(
            0,
            (ancho_label - nuevo_ancho) // 2
        )
        self.offset_video_y = max(
            0,
            (alto_label - nuevo_alto) // 2
        )

        imagen_ctk = ctk.CTkImage(
            light_image=imagen_pil,
            dark_image=imagen_pil,
            size=(nuevo_ancho, nuevo_alto)
        )

        self.label_video.configure(
            image=imagen_ctk,
            text=""
        )

        self.label_video.image = imagen_ctk

    def capturar_master(self):
        if self.frame_actual is None:
            return

        roi = self.obtener_roi(
            self.frame_actual
        )

        if roi.size == 0:
            return

        guardado = cv2.imwrite(
            self.ruta_master,
            roi
        )

        if not guardado:
            self.label_master.configure(
                text="Error al guardar MASTER",
                text_color="#FF6B6B"
            )
            return

        self.imagen_master = roi.copy()

        self.label_master.configure(
            text="MASTER: Capturada correctamente",
            text_color="#6FE3A1"
        )

        self.label_resultado.configure(
            text="MASTER OK",
            fg_color="#315A77"
        )

        self.label_similitud.configure(
            text="Similitud: --- %"
        )

    def alternar_comparacion(self):
        if self.imagen_master is None:
            self.label_resultado.configure(
                text="SIN MASTER",
                fg_color="#A66A00"
            )
            return

        self.comparacion_activa = (
            not self.comparacion_activa
        )

        if self.comparacion_activa:
            self.boton_comparar.configure(
                text="DETENER COMPARACIÓN",
                fg_color="#B3261E",
                hover_color="#871B16"
            )
        else:
            self.boton_comparar.configure(
                text="INICIAR COMPARACIÓN",
                fg_color="#218C4A",
                hover_color="#176A38"
            )

            self.label_resultado.configure(
                text="DETENIDO",
                fg_color="#555555"
            )

            self.label_similitud.configure(
                text="Similitud: --- %"
            )

    def guardar_resultado(self):
        if self.frame_actual is None:
            return

        fecha = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        resultado = self.label_resultado.cget(
            "text"
        )

        nombre_archivo = (
            f"{resultado}_{fecha}.jpg"
        )

        ruta = os.path.join(
            self.carpeta_resultados,
            nombre_archivo
        )

        cv2.imwrite(
            ruta,
            self.frame_actual
        )

    def salir(self):
        self.comparacion_activa = False

        if self.actualizacion_programada is not None:
            try:
                self.ventana.after_cancel(
                    self.actualizacion_programada
                )
            except Exception:
                pass

        if self.camara is not None:
            self.camara.release()

        cv2.destroyAllWindows()
        self.ventana.destroy()

    def ejecutar(self):
        self.ventana.mainloop()


if __name__ == "__main__":
    aplicacion = CamaraVisionMaster()
    aplicacion.ejecutar()
