import configparser
import os
import shutil
from datetime import datetime

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, simpledialog
from skimage.metrics import structural_similarity


class CamaraVision:
    """Aplicación de inspección visual multiprograma con cantidad dinámica de ROIs."""

    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.inicializada = False
        self.ventana = ctk.CTk()
        self.ventana.title("Vision sensor")
        self.ventana.geometry("1280x720")
        # self.ventana.minsize(1280, 760)
        self.ventana.configure(fg_color="#21233C")

        self.color_fondo = "#21233C"
        self.color_panel = "#292C47"
        self.color_borde = "#454B70"

        self.camara = None
        self.frame_actual = None
        self.actualizacion_programada = None
        self.comparacion_activa = False
        self.mostrando_fotografia = False
        self.frame_fotografia = None

        self.ancho_camara = 640
        self.alto_camara = 480
        self.indice_camara = 1
        self.clave_admin = "1234"
        self.modo_admin = False

        # Configuración general de cámara y carpeta de programas.
        self.ruta_configuracion = "config.ini"
        self.carpeta_programas = "programas"
        os.makedirs(self.carpeta_programas, exist_ok=True)

        self.programa_actual = None
        self.total_rois = 0
        self.ruta_programa = None
        self.carpeta_master = None
        self.carpeta_resultados = None
        self.rois = []
        self.roi_seleccionado = 0

        # Estado del editor de ROI.
        self.configurando_roi = False
        self.roi_arrastrando = False
        self.punto_inicio_roi = None
        self.roi_temporal = None

        # Conversión de coordenadas de pantalla a imagen de cámara.
        self.escala_video = 1.0
        self.offset_video_x = 0
        self.offset_video_y = 0
        self.ancho_video_mostrado = self.ancho_camara
        self.alto_video_mostrado = self.alto_camara

        self.cargar_configuracion_general()
        if not self.seleccionar_programa_inicial():
            self.ventana.destroy()
            return

        self.cargar_programa(self.programa_actual)
        self.crear_interfaz()
        self.cargar_masters()
        self.actualizar_controles_roi()
        self.iniciar_camara()

        self.ventana.after(100, lambda: self.ventana.state("zoomed"))

        self.inicializada = True
        self.ventana.protocol("WM_DELETE_WINDOW", self.salir)
        # self.ventana.after(100, lambda: self.ventana.state("zoomed"))

    def crear_rois(self, cantidad):
        """Crea la cantidad solicitada de ROIs con parámetros predeterminados."""
        rois = []
        columnas = 4
        ancho_roi = 120
        alto_roi = 90
        separacion_x = 20
        separacion_y = 20
        inicio_x = 40
        inicio_y = 60

        for indice in range(cantidad):
            columna = indice % columnas
            renglon = (indice // columnas) % 4
            x1 = inicio_x + columna * (ancho_roi + separacion_x)
            y1 = inicio_y + renglon * (alto_roi + separacion_y)
            x2 = min(x1 + ancho_roi, self.ancho_camara - 1)
            y2 = min(y1 + alto_roi, self.alto_camara - 1)
            rois.append({
                "nombre": f"Área {indice + 1}",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "similitud_minima": 90.0,
                "sensibilidad": 70,
                "area_minima": 20,
                "area_defecto_maxima": 50,
                "porcentaje_diferente_maximo": 0.20,
                "master": None,
                "resultado": "SIN MASTER",
                "similitud": None,
                "diferencia": 0.0,
                "defectos": 0,
            })
        return rois

    def listar_programas(self):
        programas = []
        if not os.path.exists(self.carpeta_programas):
            return programas
        for nombre in sorted(os.listdir(self.carpeta_programas)):
            ruta = os.path.join(self.carpeta_programas, nombre)
            ini = os.path.join(ruta, "programa.ini")
            if os.path.isdir(ruta) and os.path.exists(ini):
                programas.append(nombre)
        return programas

    @staticmethod
    def nombre_programa_valido(nombre):
        nombre = nombre.strip()
        if not nombre:
            return False
        caracteres_invalidos = '<>:"/\\|?*'
        return not any(c in nombre for c in caracteres_invalidos)

    def crear_programa(self, nombre, cantidad_rois):
        nombre = nombre.strip()
        if not self.nombre_programa_valido(nombre):
            return False, "Nombre de programa no válido."
        try:
            cantidad_rois = int(cantidad_rois)
        except (TypeError, ValueError):
            return False, "La cantidad de ROIs debe ser numérica."
        if cantidad_rois < 1 or cantidad_rois > 50:
            return False, "La cantidad de ROIs debe estar entre 1 y 50."

        carpeta = os.path.join(self.carpeta_programas, nombre)
        if os.path.exists(carpeta):
            return False, "Ese programa ya existe."

        os.makedirs(os.path.join(carpeta, "masters"), exist_ok=True)
        os.makedirs(os.path.join(carpeta, "resultados"), exist_ok=True)

        config = configparser.ConfigParser()
        config["PROGRAMA"] = {"nombre": nombre,
                              "cantidad_rois": str(cantidad_rois)}
        rois = self.crear_rois(cantidad_rois)
        for indice, roi in enumerate(rois, start=1):
            config[f"ROI_{indice}"] = {
                "nombre": roi["nombre"],
                "x1": str(roi["x1"]), "y1": str(roi["y1"]),
                "x2": str(roi["x2"]), "y2": str(roi["y2"]),
                "similitud_minima": f'{roi["similitud_minima"]:.2f}',
                "sensibilidad": str(roi["sensibilidad"]),
                "area_minima": str(roi["area_minima"]),
                "area_defecto_maxima": f'{roi["area_defecto_maxima"]:.2f}',
                "porcentaje_diferente_maximo": f'{roi["porcentaje_diferente_maximo"]:.4f}',
            }
        with open(os.path.join(carpeta, "programa.ini"), "w", encoding="utf-8") as archivo:
            config.write(archivo)
        return True, "Programa creado correctamente."

    def solicitar_clave_admin(self, parent, accion="realizar esta acción"):
        """Solicita y valida la contraseña de administrador."""
        clave = simpledialog.askstring(
            "Acceso de administrador",
            f"Ingresa la contraseña de administrador para {accion}:",
            show="*",
            parent=parent,
        )

        if clave is None:
            return False

        if clave != self.clave_admin:
            messagebox.showerror(
                "Acceso denegado",
                "Contraseña incorrecta.",
                parent=parent,
            )
            return False

        return True

    def seleccionar_programa_inicial(self):
        """Muestra una ventana modal para seleccionar o crear un programa."""
        seleccionado = {"ok": False}
        dialogo = ctk.CTkToplevel(self.ventana)
        dialogo.title("Seleccionar programa")
        dialogo.geometry("560x520")
        dialogo.resizable(False, False)
        dialogo.configure(fg_color=self.color_fondo)
        dialogo.transient(self.ventana)
        dialogo.grab_set()

        ctk.CTkLabel(dialogo, text="PROGRAMAS DE INSPECCIÓN",
                     font=("Arial", 24, "bold")).pack(pady=(24, 6))
        ctk.CTkLabel(dialogo, text="Selecciona un programa existente o crea uno nuevo.",
                     text_color="#AEB4D0").pack(pady=(0, 18))

        frame_sel = ctk.CTkFrame(dialogo, fg_color=self.color_panel)
        frame_sel.pack(fill="x", padx=28, pady=8)
        ctk.CTkLabel(frame_sel, text="Programa existente", font=(
            "Arial", 14, "bold")).pack(anchor="w", padx=18, pady=(14, 6))
        programas = self.listar_programas()
        self.menu_programa_inicial = ctk.CTkOptionMenu(
            frame_sel, values=programas or ["-- Sin programas --"])
        self.menu_programa_inicial.pack(fill="x", padx=18, pady=(0, 12))

        def abrir_programa():
            valor = self.menu_programa_inicial.get().strip()
            if valor in self.listar_programas():
                self.programa_actual = valor
                seleccionado["ok"] = True
                dialogo.destroy()

        ctk.CTkButton(frame_sel, text="ABRIR PROGRAMA", height=40,
                      command=abrir_programa).pack(fill="x", padx=18, pady=(0, 6))

        def eliminar_programa():
            valor = self.menu_programa_inicial.get().strip()
            if valor not in self.listar_programas():
                return

            if not self.solicitar_clave_admin(dialogo, "eliminar el programa"):
                return

            confirmacion = messagebox.askyesno(
                "Eliminar programa",
                f"¿Eliminar el programa '{valor}'?\n\n"
                "Se eliminarán también sus masters y resultados guardados.",
                parent=dialogo,
            )
            if not confirmacion:
                return

            try:
                shutil.rmtree(os.path.join(self.carpeta_programas, valor))
            except OSError as error:
                messagebox.showerror(
                    "Error", f"No fue posible eliminar el programa.\n{error}", parent=dialogo
                )
                return

            nuevos = self.listar_programas()
            self.menu_programa_inicial.configure(
                values=nuevos or ["-- Sin programas --"])
            self.menu_programa_inicial.set(
                nuevos[0] if nuevos else "-- Sin programas --")

        ctk.CTkButton(
            frame_sel,
            text="ELIMINAR PROGRAMA",
            height=34,
            fg_color="#8B2E2E",
            hover_color="#6E2424",
            command=eliminar_programa,
        ).pack(fill="x", padx=18, pady=(0, 14))

        frame_nuevo = ctk.CTkFrame(dialogo, fg_color=self.color_panel)
        frame_nuevo.pack(fill="x", padx=28, pady=8)
        ctk.CTkLabel(frame_nuevo, text="Crear nuevo programa", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(14, 8))
        self.entry_nombre_programa = ctk.CTkEntry(
            frame_nuevo, placeholder_text="Ej. 428272")
        self.entry_nombre_programa.grid(
            row=1, column=0, padx=(18, 6), pady=6, sticky="ew")
        self.entry_cantidad_rois = ctk.CTkEntry(
            frame_nuevo, width=120, placeholder_text="ROIs")
        self.entry_cantidad_rois.grid(row=1, column=1, padx=(6, 18), pady=6)
        frame_nuevo.grid_columnconfigure(0, weight=1)
        label_estado = ctk.CTkLabel(frame_nuevo, text="", text_color="#FFB86C")
        label_estado.grid(row=2, column=0, columnspan=2, padx=18, pady=2)

        def crear_y_abrir():
            nombre = self.entry_nombre_programa.get()
            cantidad = self.entry_cantidad_rois.get()

            if not self.solicitar_clave_admin(dialogo, "crear un programa nuevo"):
                return

            ok, mensaje = self.crear_programa(nombre, cantidad)
            label_estado.configure(
                text=mensaje, text_color="#6FE3A1" if ok else "#FF6B6B")
            if ok:
                self.programa_actual = nombre.strip()
                seleccionado["ok"] = True
                dialogo.after(250, dialogo.destroy)

        ctk.CTkButton(frame_nuevo, text="CREAR Y ABRIR", height=40, fg_color="#218C4A", hover_color="#176A38",
                      command=crear_y_abrir).grid(row=3, column=0, columnspan=2, padx=18, pady=(6, 14), sticky="ew")

        dialogo.protocol("WM_DELETE_WINDOW", dialogo.destroy)
        self.ventana.wait_window(dialogo)
        return seleccionado["ok"]

    def cargar_programa(self, nombre_programa):
        self.programa_actual = nombre_programa
        carpeta = os.path.join(self.carpeta_programas, nombre_programa)
        self.ruta_programa = os.path.join(carpeta, "programa.ini")
        self.carpeta_master = os.path.join(carpeta, "masters")
        self.carpeta_resultados = os.path.join(carpeta, "resultados")
        os.makedirs(self.carpeta_master, exist_ok=True)
        os.makedirs(self.carpeta_resultados, exist_ok=True)

        config = configparser.ConfigParser()
        config.read(self.ruta_programa, encoding="utf-8")
        self.total_rois = config.getint(
            "PROGRAMA", "cantidad_rois", fallback=1)
        self.total_rois = max(1, min(50, self.total_rois))
        self.rois = self.crear_rois(self.total_rois)
        self.roi_seleccionado = 0

        for indice, roi in enumerate(self.rois, start=1):
            seccion = f"ROI_{indice}"
            if not config.has_section(seccion):
                continue
            roi["nombre"] = config.get(
                seccion, "nombre", fallback=roi["nombre"])
            roi["x1"] = config.getint(seccion, "x1", fallback=roi["x1"])
            roi["y1"] = config.getint(seccion, "y1", fallback=roi["y1"])
            roi["x2"] = config.getint(seccion, "x2", fallback=roi["x2"])
            roi["y2"] = config.getint(seccion, "y2", fallback=roi["y2"])
            roi["similitud_minima"] = config.getfloat(
                seccion, "similitud_minima", fallback=roi["similitud_minima"])
            roi["sensibilidad"] = config.getint(
                seccion, "sensibilidad", fallback=roi["sensibilidad"])
            roi["area_minima"] = config.getint(
                seccion, "area_minima", fallback=roi["area_minima"])
            roi["area_defecto_maxima"] = config.getfloat(
                seccion, "area_defecto_maxima", fallback=roi["area_defecto_maxima"])
            roi["porcentaje_diferente_maximo"] = config.getfloat(
                seccion, "porcentaje_diferente_maximo", fallback=roi["porcentaje_diferente_maximo"])

    def crear_interfaz(self):
        self.ventana.grid_columnconfigure(0, weight=1)
        self.ventana.grid_rowconfigure(1, weight=1)

        encabezado = ctk.CTkFrame(
            self.ventana,
            height=50,
            fg_color=self.color_panel,
            corner_radius=0,
        )
        encabezado.grid(row=0, column=0, sticky="ew")
        encabezado.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            encabezado,
            text="Inspection Sensor",
            font=("Arial", 20, "bold"),
            text_color="white",
        ).grid(row=0, column=0, padx=25, pady=5, sticky="w")

        ctk.CTkLabel(
            encabezado,
            text=f"Emerson Inspection   |   Programa: {self.programa_actual}   |   ROIs: {self.total_rois}",
            font=("Arial", 12),
            text_color="#AEB4D0",
        ).grid(row=1, column=0, padx=25, pady=(0, 5), sticky="w")

        contenido = ctk.CTkFrame(self.ventana, fg_color="transparent")
        contenido.grid(row=1, column=0, padx=18, pady=18, sticky="nsew")
        contenido.grid_columnconfigure(0, weight=4)
        contenido.grid_columnconfigure(1, weight=0)
        contenido.grid_rowconfigure(0, weight=1)

        self.frame_camara = ctk.CTkFrame(
            contenido,
            fg_color="#11131F",
            border_width=2,
            border_color=self.color_borde,
            corner_radius=12,
        )
        self.frame_camara.grid(row=0, column=0, padx=(0, 15), sticky="nsew")
        self.frame_camara.grid_columnconfigure(0, weight=1)
        self.frame_camara.grid_rowconfigure(0, weight=1)

        # Canvas real para que las coordenadas del mouse coincidan exactamente
        # con la posición donde se dibuja la imagen.
        self.canvas_video = tk.Canvas(
            self.frame_camara,
            bg="#11131F",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas_video.grid(row=0, column=0, padx=10,
                               pady=10, sticky="nsew")
        self.canvas_video.bind("<ButtonPress-1>", self.iniciar_dibujo_roi)
        self.canvas_video.bind("<B1-Motion>", self.actualizar_dibujo_roi)
        self.canvas_video.bind("<ButtonRelease-1>", self.finalizar_dibujo_roi)

        self.id_imagen_canvas = None
        self.imagen_tk_actual = None

        panel = ctk.CTkScrollableFrame(
            contenido,
            width=330,
            fg_color=self.color_panel,
            border_width=1,
            border_color=self.color_borde,
            corner_radius=12,
        )
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_columnconfigure(0, weight=1)

        self.label_resultado_general = ctk.CTkLabel(
            panel,
            text="En espera",
            height=60,
            corner_radius=10,
            fg_color="#555555",
            font=("Arial", 23, "bold"),
            text_color="white",
        )
        self.label_resultado_general.grid(
            row=1, column=0, padx=18, pady=0, sticky="ew")

        self.labels_resultados = []
        frame_resultados = ctk.CTkFrame(
            panel, fg_color="#252842", corner_radius=10)
        frame_resultados.grid(row=2, column=0, padx=18, pady=5, sticky="ew")
        frame_resultados.grid_columnconfigure(0, weight=1)

        for indice in range(self.total_rois):
            etiqueta = ctk.CTkLabel(
                frame_resultados,
                text=f"ROI {indice + 1}: SIN MASTER",
                anchor="w",
                font=("Arial", 10, "bold"),
                text_color="#FFB86C",
            )
            etiqueta.grid(row=indice, column=0, padx=12, pady=2, sticky="ew")
            self.labels_resultados.append(etiqueta)

        ctk.CTkLabel(
            panel,
            text="ROI a configurar",
            font=("Arial", 12, "bold"),
            text_color="#AEB4D0",
        ).grid(row=3, column=0, padx=18, pady=5, sticky="w")

        self.menu_roi = ctk.CTkOptionMenu(
            panel,
            values=[f"ROI {i}" for i in range(1, self.total_rois + 1)],
            command=self.cambiar_roi_seleccionado,
        )
        self.menu_roi.grid(row=4, column=0, padx=18, pady=5, sticky="ew")

        self.label_coordenadas = ctk.CTkLabel(
            panel,
            text="X1: ---  Y1: ---  X2: ---  Y2: ---",
            font=("Arial", 12),
            text_color="#AEB4D0",
        )
        self.label_coordenadas.grid(
            row=5, column=0, padx=18, pady=4, sticky="ew")

        self.boton_admin = ctk.CTkButton(
            panel,
            text="MODO ADMINISTRADOR",
            height=42,
            font=("Arial", 13, "bold"),
            fg_color="#7A5C00",
            hover_color="#5F4700",
            command=self.alternar_modo_admin,
        )
        self.boton_admin.grid(row=6, column=0, padx=18,
                              pady=(8, 6), sticky="ew")

        self.boton_configurar = ctk.CTkButton(
            panel,
            text="CONFIGURAR ÁREA CON MOUSE",
            height=42,
            font=("Arial", 13, "bold"),
            command=self.activar_configuracion_roi,
        )
        self.boton_configurar.grid(
            row=7, column=0, padx=18, pady=6, sticky="ew")

        self.boton_master = ctk.CTkButton(
            panel,
            text="CAPTURAR MASTER DEL ROI",
            height=42,
            font=("Arial", 13, "bold"),
            fg_color="#315A77",
            hover_color="#26465D",
            command=self.capturar_master_roi,
        )
        self.boton_master.grid(row=8, column=0, padx=18, pady=6, sticky="ew")

        self.boton_todos_masters = ctk.CTkButton(
            panel,
            text=f"CAPTURAR LOS {self.total_rois} MASTERS",
            height=42,
            font=("Arial", 13, "bold"),
            fg_color="#6C4F8A",
            hover_color="#533D6B",
            command=self.capturar_todos_los_masters,
        )
        self.boton_todos_masters.grid(
            row=9, column=0, padx=18, pady=6, sticky="ew")

        ctk.CTkLabel(
            panel,
            text="Similitud mínima",
            font=("Arial", 14, "bold"),
            text_color="#AEB4D0",
        ).grid(row=10, column=0, padx=18, pady=(16, 3), sticky="w")

        self.label_similitud_config = ctk.CTkLabel(
            panel,
            text="90.0 %",
            font=("Arial", 16, "bold"),
            text_color="white",
        )
        self.label_similitud_config.grid(row=11, column=0, padx=18, pady=2)

        self.slider_similitud = ctk.CTkSlider(
            panel,
            from_=50,
            to=100,
            number_of_steps=100,
            command=self.cambiar_similitud_roi,
        )
        self.slider_similitud.grid(
            row=12, column=0, padx=18, pady=(2, 8), sticky="ew")

        ctk.CTkLabel(
            panel,
            text="Sensibilidad de diferencias",
            font=("Arial", 14, "bold"),
            text_color="#AEB4D0",
        ).grid(row=13, column=0, padx=18, pady=(10, 3), sticky="w")

        self.label_sensibilidad = ctk.CTkLabel(
            panel,
            text="70 %",
            font=("Arial", 16, "bold"),
            text_color="white",
        )
        self.label_sensibilidad.grid(row=14, column=0, padx=18, pady=2)

        self.slider_sensibilidad = ctk.CTkSlider(
            panel,
            from_=1,
            to=100,
            number_of_steps=99,
            command=self.cambiar_sensibilidad_roi,
        )
        self.slider_sensibilidad.grid(
            row=15, column=0, padx=18, pady=(2, 4), sticky="ew")

        self.label_ayuda_sensibilidad = ctk.CTkLabel(
            panel,
            text="Mayor sensibilidad = detecta diferencias más pequeñas.",
            font=("Arial", 11),
            text_color="#AEB4D0",
            wraplength=280,
        )
        self.label_ayuda_sensibilidad.grid(
            row=16, column=0, padx=18, pady=(0, 10))

        self.boton_comparar = ctk.CTkButton(
            panel,
            text="VER EN VIVO",
            height=45,
            font=("Arial", 14, "bold"),
            fg_color="#218C4A",
            hover_color="#176A38",
            command=self.alternar_comparacion,
        )
        self.boton_comparar.grid(
            row=17, column=0, padx=18, pady=(12, 6), sticky="ew")

        self.boton_guardar = ctk.CTkButton(
            panel,
            text="INICIAR\nPRUEBA",
            height=80,
            font=("Arial", 30, "bold"),
            fg_color="#8A6D1D",
            hover_color="#6C5515",
            command=self.guardar_resultado,
        )
        self.boton_guardar.grid(row=18, column=0, padx=18, pady=6, sticky="ew")

        self.actualizar_estado_modo_admin()

    def alternar_modo_admin(self):
        """Activa o desactiva los controles que modifican la receta de inspección."""
        if self.modo_admin:
            self.modo_admin = False
            if self.configurando_roi:
                self.cancelar_configuracion_roi()
            self.actualizar_estado_modo_admin()
            self.label_resultado_general.configure(
                text="MODO OPERADOR",
                fg_color="#555555",
            )
            return

        if not self.solicitar_clave_admin(self.ventana, "activar el modo administrador"):
            return

        self.modo_admin = True
        self.actualizar_estado_modo_admin()
        self.label_resultado_general.configure(
            text="MODO ADMINISTRADOR",
            fg_color="#7A5C00",
        )

    def actualizar_estado_modo_admin(self):
        """Bloquea la edición de la receta mientras la aplicación está en modo operador."""
        if not hasattr(self, "boton_admin"):
            return

        estado = "normal" if self.modo_admin else "disabled"

        for control in (
            self.boton_configurar,
            self.boton_master,
            self.boton_todos_masters,
            self.slider_similitud,
            self.slider_sensibilidad,
        ):
            control.configure(state=estado)

        if self.modo_admin:
            self.boton_admin.configure(
                text="SALIR DE MODO ADMIN",
                fg_color="#B3261E",
                hover_color="#871B16",
            )
        else:
            self.boton_admin.configure(
                text="MODO ADMINISTRADOR",
                fg_color="#7A5C00",
                hover_color="#5F4700",
            )

    def cargar_configuracion_general(self):
        config = configparser.ConfigParser()

        if os.path.exists(self.ruta_configuracion):
            try:
                config.read(self.ruta_configuracion, encoding="utf-8")
            except configparser.Error as error:
                print(f"No fue posible cargar config.ini: {error}")
                config = configparser.ConfigParser()

        if config.has_section("CAMARA"):
            try:
                self.indice_camara = config.getint(
                    "CAMARA", "indice", fallback=self.indice_camara
                )
                self.ancho_camara = config.getint(
                    "CAMARA", "ancho", fallback=self.ancho_camara
                )
                self.alto_camara = config.getint(
                    "CAMARA", "alto", fallback=self.alto_camara
                )
            except ValueError as error:
                print(f"Configuración de cámara no válida: {error}")

        if config.has_section("ADMIN"):
            self.clave_admin = config.get(
                "ADMIN", "clave", fallback=self.clave_admin
            ).strip() or "1234"

        # Garantiza que config.ini siempre tenga CAMARA y ADMIN sin borrar
        # los valores que ya existían.
        if not config.has_section("CAMARA"):
            config["CAMARA"] = {}
        config["CAMARA"]["indice"] = str(self.indice_camara)
        config["CAMARA"]["ancho"] = str(self.ancho_camara)
        config["CAMARA"]["alto"] = str(self.alto_camara)

        if not config.has_section("ADMIN"):
            config["ADMIN"] = {}
        if not config["ADMIN"].get("clave", "").strip():
            config["ADMIN"]["clave"] = self.clave_admin

        with open(self.ruta_configuracion, "w", encoding="utf-8") as archivo:
            config.write(archivo)

    def guardar_configuracion_general(self):
        config = configparser.ConfigParser()
        if os.path.exists(self.ruta_configuracion):
            try:
                config.read(self.ruta_configuracion, encoding="utf-8")
            except configparser.Error:
                config = configparser.ConfigParser()

        if not config.has_section("CAMARA"):
            config["CAMARA"] = {}
        config["CAMARA"].update({
            "indice": str(self.indice_camara),
            "ancho": str(self.ancho_camara),
            "alto": str(self.alto_camara),
        })

        if not config.has_section("ADMIN"):
            config["ADMIN"] = {}
        config["ADMIN"]["clave"] = self.clave_admin

        with open(self.ruta_configuracion, "w", encoding="utf-8") as archivo:
            config.write(archivo)

    def guardar_configuracion(self):
        """Guarda la configuración del programa actualmente cargado."""
        if not self.programa_actual or not self.ruta_programa:
            return
        config = configparser.ConfigParser()
        config["PROGRAMA"] = {
            "nombre": self.programa_actual,
            "cantidad_rois": str(self.total_rois),
        }
        for indice, roi in enumerate(self.rois, start=1):
            config[f"ROI_{indice}"] = {
                "nombre": roi["nombre"],
                "x1": str(roi["x1"]), "y1": str(roi["y1"]),
                "x2": str(roi["x2"]), "y2": str(roi["y2"]),
                "similitud_minima": f'{roi["similitud_minima"]:.2f}',
                "sensibilidad": str(int(roi["sensibilidad"])),
                "area_minima": str(int(roi["area_minima"])),
                "area_defecto_maxima": f'{roi["area_defecto_maxima"]:.2f}',
                "porcentaje_diferente_maximo": f'{roi["porcentaje_diferente_maximo"]:.4f}',
            }
        with open(self.ruta_programa, "w", encoding="utf-8") as archivo:
            config.write(archivo)

    def ruta_master_roi(self, indice):
        return os.path.join(self.carpeta_master, f"roi_{indice + 1}.jpg")

    def cargar_masters(self):
        for indice, roi in enumerate(self.rois):
            ruta = self.ruta_master_roi(indice)
            imagen = cv2.imread(ruta) if os.path.exists(ruta) else None
            roi["master"] = imagen
            roi["resultado"] = "LISTO" if imagen is not None else "SIN MASTER"

    def iniciar_camara(self):
        self.camara = cv2.VideoCapture(self.indice_camara, cv2.CAP_DSHOW)

        if not self.camara.isOpened():
            self.canvas_video.delete("all")
            self.canvas_video.create_text(
                320, 240,
                text=f"No fue posible abrir la cámara {self.indice_camara}.",
                fill="#AEB4D0",
                font=("Arial", 18, "bold"),
            )
            return

        self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho_camara)
        self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto_camara)
        self.camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # self.camara.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.actualizar_video()

    def cambiar_roi_seleccionado(self, valor):
        try:
            self.roi_seleccionado = int(valor.split()[-1]) - 1
        except (ValueError, IndexError):
            self.roi_seleccionado = 0
        self.actualizar_controles_roi()

    def actualizar_controles_roi(self):
        roi = self.rois[self.roi_seleccionado]
        self.menu_roi.set(f"ROI {self.roi_seleccionado + 1}")
        self.slider_similitud.set(roi["similitud_minima"])
        self.slider_sensibilidad.set(roi["sensibilidad"])
        self.label_similitud_config.configure(
            text=f'{roi["similitud_minima"]:.1f} %')
        self.label_sensibilidad.configure(text=f'{int(roi["sensibilidad"])} %')
        self.label_coordenadas.configure(
            text=(
                f'X1: {roi["x1"]}  Y1: {roi["y1"]}  '
                f'X2: {roi["x2"]}  Y2: {roi["y2"]}'
            )
        )

    def cambiar_similitud_roi(self, valor):

        if not self.modo_admin:
            return
        roi = self.rois[self.roi_seleccionado]
        roi["similitud_minima"] = float(valor)
        self.label_similitud_config.configure(text=f"{float(valor):.1f} %")
        self.guardar_configuracion()

    def cambiar_sensibilidad_roi(self, valor):

        if not self.modo_admin:
            return
        roi = self.rois[self.roi_seleccionado]
        roi["sensibilidad"] = int(round(float(valor)))
        self.label_sensibilidad.configure(text=f'{roi["sensibilidad"]} %')
        self.guardar_configuracion()

    def activar_configuracion_roi(self):

        if not self.modo_admin:
            return
        self.comparacion_activa = False
        self.configurando_roi = True
        self.roi_arrastrando = False
        self.punto_inicio_roi = None
        self.roi_temporal = None
        self.boton_comparar.configure(
            text="VER EN VIVO",
            fg_color="#218C4A",
            hover_color="#176A38",
        )
        self.boton_configurar.configure(
            text=f"DIBUJA EL ROI {self.roi_seleccionado + 1}",
            fg_color="#A66A00",
            hover_color="#815200",
        )
        self.label_resultado_general.configure(
            text="CONFIGURANDO ROI", fg_color="#A66A00")

    def coordenada_pantalla_a_frame(self, x, y):
        if self.frame_actual is None or self.escala_video <= 0:
            return None

        x_imagen = (x - self.offset_video_x) / self.escala_video
        y_imagen = (y - self.offset_video_y) / self.escala_video

        alto, ancho = self.frame_actual.shape[:2]
        if x_imagen < 0 or y_imagen < 0 or x_imagen >= ancho or y_imagen >= alto:
            return None

        return int(x_imagen), int(y_imagen)

    def iniciar_dibujo_roi(self, evento):
        if not self.configurando_roi:
            return

        punto = self.coordenada_pantalla_a_frame(evento.x, evento.y)
        if punto is None:
            return

        self.roi_arrastrando = True
        self.punto_inicio_roi = punto
        self.roi_temporal = (*punto, *punto)

    def actualizar_dibujo_roi(self, evento):
        if not self.configurando_roi or not self.roi_arrastrando:
            return

        punto = self.coordenada_pantalla_a_frame(evento.x, evento.y)
        if punto is None:
            return

        x_inicial, y_inicial = self.punto_inicio_roi
        x_actual, y_actual = punto
        self.roi_temporal = (
            min(x_inicial, x_actual),
            min(y_inicial, y_actual),
            max(x_inicial, x_actual),
            max(y_inicial, y_actual),
        )

    def finalizar_dibujo_roi(self, evento):
        if not self.configurando_roi or not self.roi_arrastrando:
            return

        punto = self.coordenada_pantalla_a_frame(evento.x, evento.y)
        self.roi_arrastrando = False

        if punto is None or self.punto_inicio_roi is None:
            self.cancelar_configuracion_roi()
            return

        x_inicial, y_inicial = self.punto_inicio_roi
        x_final, y_final = punto
        x1, x2 = sorted((x_inicial, x_final))
        y1, y2 = sorted((y_inicial, y_final))

        if x2 - x1 < 20 or y2 - y1 < 20:
            self.label_resultado_general.configure(
                text="ROI MUY PEQUEÑO", fg_color="#B3261E")
            self.cancelar_configuracion_roi()
            return

        roi = self.rois[self.roi_seleccionado]
        roi.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        # Al cambiar el tamaño del ROI, la imagen MASTER anterior deja de ser válida.
        roi["master"] = None
        roi["resultado"] = "SIN MASTER"
        ruta = self.ruta_master_roi(self.roi_seleccionado)
        if os.path.exists(ruta):
            try:
                os.remove(ruta)
            except OSError:
                pass

        self.guardar_configuracion()
        self.configurando_roi = False
        self.roi_temporal = None
        self.punto_inicio_roi = None
        self.boton_configurar.configure(
            text="CONFIGURAR ÁREA CON MOUSE",
            fg_color=["#3B8ED0", "#1F6AA5"],
            hover_color=["#36719F", "#144870"],
        )
        self.label_resultado_general.configure(
            text="ROI GUARDADO", fg_color="#315A77")
        self.actualizar_controles_roi()
        self.actualizar_etiquetas_resultados()

    def cancelar_configuracion_roi(self):
        self.configurando_roi = False
        self.roi_arrastrando = False
        self.punto_inicio_roi = None
        self.roi_temporal = None
        self.boton_configurar.configure(
            text="CONFIGURAR ÁREA CON MOUSE",
            fg_color=["#3B8ED0", "#1F6AA5"],
            hover_color=["#36719F", "#144870"],
        )

    def obtener_roi(self, frame, roi):
        alto, ancho = frame.shape[:2]
        x1 = max(0, min(int(roi["x1"]), ancho - 1))
        y1 = max(0, min(int(roi["y1"]), alto - 1))
        x2 = max(x1 + 1, min(int(roi["x2"]), ancho))
        y2 = max(y1 + 1, min(int(roi["y2"]), alto))
        return frame[y1:y2, x1:x2]

    def capturar_master_roi(self):

        if not self.modo_admin:
            return
        if self.frame_actual is None:
            return

        roi_config = self.rois[self.roi_seleccionado]
        imagen_roi = self.obtener_roi(self.frame_actual, roi_config)
        if imagen_roi.size == 0:
            return

        ruta = self.ruta_master_roi(self.roi_seleccionado)
        if not cv2.imwrite(ruta, imagen_roi):
            self.label_resultado_general.configure(
                text="ERROR AL GUARDAR", fg_color="#B3261E")
            return

        roi_config["master"] = imagen_roi.copy()
        roi_config["resultado"] = "LISTO"
        self.label_resultado_general.configure(
            text="MASTER GUARDADA", fg_color="#315A77")
        self.actualizar_etiquetas_resultados()

    def capturar_todos_los_masters(self):

        if not self.modo_admin:
            return
        if self.frame_actual is None:
            return

        guardados = 0
        for indice, roi_config in enumerate(self.rois):
            imagen_roi = self.obtener_roi(self.frame_actual, roi_config)
            if imagen_roi.size == 0:
                continue
            if cv2.imwrite(self.ruta_master_roi(indice), imagen_roi):
                roi_config["master"] = imagen_roi.copy()
                roi_config["resultado"] = "LISTO"
                guardados += 1

        self.label_resultado_general.configure(
            text=f"{guardados} MASTERS GUARDADAS",
            fg_color="#315A77" if guardados == self.total_rois else "#A66A00",
        )
        self.actualizar_etiquetas_resultados()

    @staticmethod
    def preparar_imagen(imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gris, (9, 9), 0)

    @staticmethod
    def calcular_umbral_diferencia(sensibilidad):
        """
        Convierte sensibilidad 1-100 a umbral binario.
        Mayor sensibilidad produce un umbral menor y detecta cambios más pequeños.
        """
        sensibilidad = max(1, min(100, int(sensibilidad)))
        # 109 a 30 aproximadamente.
        return int(round(110 - (sensibilidad * 0.8)))

    def comparar_roi_con_master(self, imagen_actual, roi_config):
        master = roi_config["master"]
        if master is None or imagen_actual.size == 0:
            return None

        imagen_actual = cv2.resize(
            imagen_actual, (master.shape[1], master.shape[0]))
        gris_master = self.preparar_imagen(master)
        gris_actual = self.preparar_imagen(imagen_actual)

        similitud, mapa_similitud = structural_similarity(
            gris_master,
            gris_actual,
            full=True,
            data_range=255,
        )
        porcentaje_similitud = float(similitud * 100)

        mapa_similitud = np.clip(mapa_similitud * 255, 0, 255).astype("uint8")
        mapa_diferencias = cv2.bitwise_not(mapa_similitud)
        umbral_diferencia = self.calcular_umbral_diferencia(
            roi_config["sensibilidad"])

        _, mascara = cv2.threshold(
            mapa_diferencias,
            umbral_diferencia,
            255,
            cv2.THRESH_BINARY,
        )

        kernel = np.ones((3, 3), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        imagen_resultado = imagen_actual.copy()
        area_total_diferente = 0.0
        area_defecto_mayor = 0.0
        cantidad_defectos = 0

        for contorno in contornos:
            area = float(cv2.contourArea(contorno))
            if area < roi_config["area_minima"]:
                continue

            cantidad_defectos += 1
            area_total_diferente += area
            area_defecto_mayor = max(area_defecto_mayor, area)
            x, y, ancho, alto = cv2.boundingRect(contorno)
            cv2.rectangle(imagen_resultado, (x, y),
                          (x + ancho, y + alto), (0, 0, 255), 2)

        area_roi = max(1, master.shape[0] * master.shape[1])
        porcentaje_diferente = (area_total_diferente / area_roi) * 100

        aprobado = (
            porcentaje_similitud >= roi_config["similitud_minima"]
            and area_defecto_mayor < roi_config["area_defecto_maxima"]
            and porcentaje_diferente < roi_config["porcentaje_diferente_maximo"]
        )

        return {
            "aprobado": aprobado,
            "similitud": porcentaje_similitud,
            "diferencia": porcentaje_diferente,
            "defectos": cantidad_defectos,
            "area_defecto_mayor": area_defecto_mayor,
            "imagen_resultado": imagen_resultado,
        }

    def procesar_frame_inspeccion(self, frame):
        """
        Compara una fotografía completa contra los MASTER del programa seleccionado.

        Devuelve la imagen anotada y el resultado general. Esta función se
        utiliza tanto para el modo en vivo como para la inspección por foto.
        """
        frame_resultado = frame.copy()
        resultados_validos = []

        for indice, roi in enumerate(self.rois):
            x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
            color = (0, 255, 255)

            if roi["master"] is None:
                roi["resultado"] = "SIN MASTER"
                roi["similitud"] = None
                roi["diferencia"] = 0.0
                roi["defectos"] = 0
            else:
                imagen_roi = self.obtener_roi(frame, roi)
                resultado = self.comparar_roi_con_master(imagen_roi, roi)

                if resultado is None:
                    roi["resultado"] = "ERROR"
                    roi["similitud"] = None
                    roi["diferencia"] = 0.0
                    roi["defectos"] = 0
                    color = (0, 165, 255)
                else:
                    aprobado = bool(resultado["aprobado"])
                    roi["resultado"] = "PASS" if aprobado else "FAIL"
                    roi["similitud"] = resultado["similitud"]
                    roi["diferencia"] = resultado["diferencia"]
                    roi["defectos"] = resultado["defectos"]
                    resultados_validos.append(aprobado)
                    color = (0, 255, 0) if aprobado else (0, 0, 255)

                    imagen_resultado = cv2.resize(
                        resultado["imagen_resultado"],
                        (max(1, x2 - x1), max(1, y2 - y1)),
                    )
                    frame_resultado[y1:y2, x1:x2] = imagen_resultado

            grosor = 4 if indice == self.roi_seleccionado else 2
            cv2.rectangle(frame_resultado, (x1, y1), (x2, y2), color, grosor)

            texto = f'ROI {indice + 1}: {roi["resultado"]}'
            cv2.putText(
                frame_resultado,
                texto,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        masters_cargadas = sum(roi["master"] is not None for roi in self.rois)

        if masters_cargadas < self.total_rois:
            resultado_general = (
                f"FALTAN {self.total_rois - masters_cargadas} MASTERS"
            )
            color_general = "#A66A00"
        elif len(resultados_validos) == self.total_rois and all(resultados_validos):
            resultado_general = "PASS"
            color_general = "#218C4A"
        else:
            resultado_general = "FAIL"
            color_general = "#B3261E"

        self.label_resultado_general.configure(
            text=resultado_general,
            fg_color=color_general,
        )
        self.actualizar_etiquetas_resultados()

        return frame_resultado, resultado_general

    def actualizar_video(self):
        if self.camara is None:
            return

        correcto, frame = self.camara.read()
        if not correcto or frame is None:
            self.actualizacion_programada = self.ventana.after(
                30, self.actualizar_video
            )
            return

        self.frame_actual = frame.copy()

        # Cuando se tomó una fotografía, se mantiene congelada hasta volver
        # a activar el modo en vivo.
        if self.mostrando_fotografia and self.frame_fotografia is not None:
            frame_mostrar = self.frame_fotografia.copy()
        elif self.comparacion_activa:
            frame_mostrar, _ = self.procesar_frame_inspeccion(frame)
        else:
            frame_mostrar = frame.copy()

            for indice, roi in enumerate(self.rois):
                x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
                color = (0, 255, 255)
                grosor = 4 if indice == self.roi_seleccionado else 2

                cv2.rectangle(
                    frame_mostrar,
                    (x1, y1),
                    (x2, y2),
                    color,
                    grosor,
                )
                cv2.putText(
                    frame_mostrar,
                    f"ROI {indice + 1}",
                    (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        if self.roi_temporal is not None and not self.mostrando_fotografia:
            tx1, ty1, tx2, ty2 = self.roi_temporal
            cv2.rectangle(
                frame_mostrar,
                (tx1, ty1),
                (tx2, ty2),
                (255, 255, 0),
                2,
            )

        self.mostrar_frame(frame_mostrar)
        self.actualizacion_programada = self.ventana.after(
            30, self.actualizar_video
        )

    def actualizar_etiquetas_resultados(self):
        for indice, (roi, etiqueta) in enumerate(zip(self.rois, self.labels_resultados), start=1):
            if roi["similitud"] is None or not self.comparacion_activa:
                texto = f'ROI {indice}: {roi["resultado"]}'
            else:
                texto = (
                    f'ROI {indice}: {roi["resultado"]} | '
                    f'{roi["similitud"]:.2f}% | D:{roi["diferencia"]:.3f}%'
                )

            if roi["resultado"] == "PASS":
                color = "#6FE3A1"
            elif roi["resultado"] == "FAIL":
                color = "#FF6B6B"
            elif roi["resultado"] == "SIN MASTER":
                color = "#FFB86C"
            else:
                color = "#AEB4D0"

            etiqueta.configure(text=texto, text_color=color)

    def mostrar_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen_pil = Image.fromarray(frame_rgb)

        # Tamaño real del Canvas. No se usan valores mínimos artificiales porque
        # provocarían una escala distinta a la imagen realmente visible.
        ancho_disponible = self.canvas_video.winfo_width()
        alto_disponible = self.canvas_video.winfo_height()

        if ancho_disponible <= 1 or alto_disponible <= 1:
            self.ventana.after(20, lambda: self.mostrar_frame(frame))
            return

        ancho_original, alto_original = imagen_pil.size
        escala = min(
            ancho_disponible / ancho_original,
            alto_disponible / alto_original,
        )

        nuevo_ancho = max(1, round(ancho_original * escala))
        nuevo_alto = max(1, round(alto_original * escala))

        # La imagen se coloca con anchor="nw" exactamente en estos offsets.
        self.escala_video = escala
        self.ancho_video_mostrado = nuevo_ancho
        self.alto_video_mostrado = nuevo_alto
        self.offset_video_x = (ancho_disponible - nuevo_ancho) / 2
        self.offset_video_y = (alto_disponible - nuevo_alto) / 2

        imagen_pil = imagen_pil.resize(
            (nuevo_ancho, nuevo_alto),
            Image.Resampling.LANCZOS,
        )
        self.imagen_tk_actual = ImageTk.PhotoImage(imagen_pil)

        if self.id_imagen_canvas is None:
            self.id_imagen_canvas = self.canvas_video.create_image(
                self.offset_video_x,
                self.offset_video_y,
                anchor="nw",
                image=self.imagen_tk_actual,
            )
        else:
            self.canvas_video.coords(
                self.id_imagen_canvas,
                self.offset_video_x,
                self.offset_video_y,
            )
            self.canvas_video.itemconfigure(
                self.id_imagen_canvas,
                image=self.imagen_tk_actual,
            )

    def alternar_comparacion(self):
        """Activa o detiene la comparación continua en vivo."""
        if self.configurando_roi:
            self.cancelar_configuracion_roi()

        if not any(roi["master"] is not None for roi in self.rois):
            self.label_resultado_general.configure(
                text="SIN MASTERS",
                fg_color="#A66A00",
            )
            return

        # Salir de la fotografía congelada al entrar al modo en vivo.
        self.mostrando_fotografia = False
        self.frame_fotografia = None
        self.comparacion_activa = not self.comparacion_activa

        if self.comparacion_activa:
            self.boton_comparar.configure(
                text="DETENER MODO EN VIVO",
                fg_color="#B3261E",
                hover_color="#871B16",
            )
        else:
            self.boton_comparar.configure(
                text="VER EN VIVO",
                fg_color="#218C4A",
                hover_color="#176A38",
            )
            self.label_resultado_general.configure(
                text="EN ESPERA",
                fg_color="#555555",
            )
            for roi in self.rois:
                roi["resultado"] = (
                    "LISTO" if roi["master"] is not None else "SIN MASTER"
                )
                roi["similitud"] = None
                roi["diferencia"] = 0.0
                roi["defectos"] = 0
            self.actualizar_etiquetas_resultados()

    def guardar_resultado(self):
        """
        Toma una fotografía nueva, realiza una sola comparación, guarda la
        evidencia y deja la imagen congelada en pantalla.
        """
        if self.camara is None or not self.camara.isOpened():
            self.label_resultado_general.configure(
                text="CÁMARA NO DISPONIBLE",
                fg_color="#B3261E",
            )
            return

        if not any(roi["master"] is not None for roi in self.rois):
            self.label_resultado_general.configure(
                text="SIN MASTERS",
                fg_color="#A66A00",
            )
            return

        if self.configurando_roi:
            self.cancelar_configuracion_roi()

        # Detener el análisis continuo para que esta inspección sea una
        # fotografía única y reproducible.
        self.comparacion_activa = False
        self.boton_comparar.configure(
            text="VER EN VIVO",
            fg_color="#218C4A",
            hover_color="#176A38",
        )

        # Vaciar algunos cuadros antiguos del búfer y tomar una imagen reciente.
        frame_capturado = None
        for _ in range(3):
            correcto, frame = self.camara.read()
            if correcto and frame is not None:
                frame_capturado = frame.copy()

        if frame_capturado is None:
            self.label_resultado_general.configure(
                text="ERROR DE CAPTURA",
                fg_color="#B3261E",
            )
            return

        self.frame_actual = frame_capturado.copy()
        frame_comparado, resultado_general = self.procesar_frame_inspeccion(
            frame_capturado
        )

        fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nombre_resultado = resultado_general.replace(" ", "_")
        ruta = os.path.join(
            self.carpeta_resultados,
            f"{nombre_resultado}_{fecha}.jpg",
        )

        # Agregar fecha, hora y resultado general a la evidencia.
        cv2.rectangle(
            frame_comparado,
            (0, 0),
            (frame_comparado.shape[1], 38),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame_comparado,
            f"{resultado_general}  {fecha}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if resultado_general == "PASS" else (0, 0, 255),
            2,
        )

        if not cv2.imwrite(ruta, frame_comparado):
            self.label_resultado_general.configure(
                text="ERROR AL GUARDAR",
                fg_color="#B3261E",
            )
            return

        # Mantener la fotografía inspeccionada visible hasta presionar
        # nuevamente VER EN VIVO.
        self.frame_fotografia = frame_comparado.copy()
        self.mostrando_fotografia = True
        self.mostrar_frame(self.frame_fotografia)

        self.label_resultado_general.configure(
            text=f"{resultado_general}",
            fg_color="#218C4A" if resultado_general == "PASS" else "#B3261E",
        )

    def salir(self):
        self.comparacion_activa = False
        if self.actualizacion_programada is not None:
            try:
                self.ventana.after_cancel(self.actualizacion_programada)
            except Exception:
                pass

        if self.camara is not None:
            self.camara.release()

        cv2.destroyAllWindows()
        self.ventana.destroy()

    def ejecutar(self):
        if self.inicializada:
            self.ventana.mainloop()


if __name__ == "__main__":
    aplicacion = CamaraVision()
    aplicacion.ejecutar()
