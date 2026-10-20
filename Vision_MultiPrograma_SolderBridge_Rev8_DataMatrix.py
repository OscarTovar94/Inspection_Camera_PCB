"""
Software de inspección visual multiprograma con cantidad dinámica de ROIs.
Permite crear programas de inspección con un número variable de ROIs (áreas de interés) y configurar parámetros de inspección para cada ROI. La aplicación captura imágenes de una cámara, permite definir ROIs con el mouse, capturar imágenes maestras (masters) y comparar imágenes en tiempo real para detectar diferencias.
Requiere Python 3.8 o superior y las siguientes librerías:
- OpenCV (cv2)
- NumPy
"""
import configparser
import os
import shutil
import re
import platform
from datetime import datetime
from tkinter import messagebox, simpledialog, ttk
import tkinter as tk
import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity  # pylint: disable=no-name-in-module

try:
    import zxingcpp  # pylint: disable=no-name-in-module
except ImportError:
    zxingcpp = None


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

        # Rutas absolutas ancladas a la ubicación de este script.
        # Así siempre se utiliza el config.ini que está junto al programa,
        # independientemente de la carpeta desde la que se ejecute Python.
        self.directorio_base = os.path.dirname(os.path.abspath(__file__))
        self.ruta_configuracion = os.path.join(
            self.directorio_base, "config.ini")
        self.carpeta_programas = os.path.join(
            self.directorio_base, "programas")
        os.makedirs(self.carpeta_programas, exist_ok=True)

        # IDs de callbacks programados por nuestra aplicación.
        self.after_zoom_id = None
        self.cerrando = False

        self.programa_actual = None
        self.total_rois = 0
        self.ruta_programa = None
        self.carpeta_master = None
        self.carpeta_resultados = None
        self.rois = []
        self.roi_seleccionado = 0
        self.max_masters = 5
        self.referencia_pcb = None
        self.area_pcb = None
        self.alineacion_info = "SIN REFERENCIA"
        self.configurando_area_pcb = False
        self.max_desplazamiento = 15.0
        self.max_rotacion = 3.0
        self.min_confianza_alineacion = 0.80

        # Estado del editor de ROI.
        self.configurando_roi = False
        self.roi_arrastrando = False
        self.punto_inicio_roi = None
        self.roi_temporal = None
        self.calibrando_pads = False
        self.pads_temporales = []
        self.ventana_mascara = None

        # Conversión de coordenadas de pantalla a imagen de cámara.
        self.escala_video = 1.0
        self.offset_video_x = 0
        self.offset_video_y = 0
        self.ancho_video_mostrado = self.ancho_camara
        self.alto_video_mostrado = self.alto_camara

        self.cargar_configuracion_general()
        if not self.seleccionar_programa_inicial():
            self.cerrando = True
            try:
                if self.ventana.winfo_exists():
                    self.ventana.destroy()
            except tk.TclError:
                pass
            return

        self.cargar_programa(self.programa_actual)

        print("[INICIO] Creando interfaz principal...")
        self.crear_interfaz()
        print("[INICIO] Interfaz principal creada.")

        self.cargar_masters()
        self.cargar_referencia_pcb()
        self.actualizar_controles_roi()

        print("[CAMARA] Inicializando cámara...")
        self.iniciar_camara()
        print("[CAMARA] Inicialización de cámara finalizada.")

        self.after_zoom_id = self.ventana.after(
            100, self.maximizar_ventana_seguro)

        self.inicializada = True
        self.ventana.protocol("WM_DELETE_WINDOW", self.salir)
        # self.ventana.after(100, lambda: self.ventana.state("zoomed"))

    def maximizar_ventana_seguro(self):
        """Maximiza la ventana sólo si Tk sigue activo."""
        try:
            if platform.system() == "Windows":
                self.ventana.state("zoomed")
            elif platform.system() == "Linux":
                self.ventana.wm_attributes("-zoomed", True)
            else:
                self.ventana.state("zoomed")
        except tk.TclError:
            pass

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
                "tipo": "INSPECCION",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "similitud_minima": 90.0,
                "sensibilidad": 70,
                "area_minima": 20,
                "area_defecto_maxima": 50,
                "porcentaje_diferente_maximo": 0.20,
                "master": None,
                "masters": [],
                "pads": [],
                "sat_max": 75,
                "val_min": 137,
                "dm_longitud": 16,
                "dm_numero_parte": "",
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
                "tipo": roi["tipo"],
                "pads": ";".join(f"{x},{y}" for x, y in roi.get("pads", [])),
                "sat_max": str(int(roi.get("sat_max", 75))),
                "val_min": str(int(roi.get("val_min", 137))),
                "dm_longitud": str(int(roi.get("dm_longitud", 16))),
                "dm_numero_parte": str(roi.get("dm_numero_parte", "")),
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
        """
        Selector inicial implementado con Tkinter nativo.

        Se evita crear/destruir una ventana CTkToplevel antes del mainloop
        principal, porque CustomTkinter programa callbacks internos de escala,
        apariencia y DPI que pueden quedar pendientes al destruir esa ventana.
        """
        seleccionado = {"ok": False}

        try:
            self.ventana.withdraw()
        except tk.TclError:
            return False

        dialogo = tk.Toplevel(self.ventana)
        dialogo.title("Seleccionar programa")
        dialogo.geometry("560x470")
        dialogo.resizable(False, False)
        dialogo.configure(bg=self.color_fondo)

        # IMPORTANTE:
        # No se usa transient(self.ventana) porque la ventana principal está
        # temporalmente oculta con withdraw(). En Windows, un Toplevel transient
        # de una ventana oculta puede quedar creado pero no mostrarse.
        dialogo.attributes("-topmost", True)

        # Centrar aproximadamente en pantalla.
        dialogo.update_idletasks()
        ancho = 560
        alto = 500
        x = max(0, (dialogo.winfo_screenwidth() - ancho) // 2)
        y = max(0, (dialogo.winfo_screenheight() - alto) // 2)
        dialogo.geometry(f"{ancho}x{alto}+{x}+{y}")
        dialogo.deiconify()
        dialogo.lift()
        dialogo.focus_force()

        def quitar_topmost():
            try:
                if dialogo.winfo_exists():
                    dialogo.attributes("-topmost", False)
            except tk.TclError:
                pass

        dialogo.after(300, quitar_topmost)

        print("[INICIO] Selector de programas visible.")

        titulo = tk.Label(
            dialogo,
            text="PROGRAMAS DE INSPECCIÓN",
            font=("Arial", 22, "bold"),
            fg="white",
            bg=self.color_fondo,
        )
        titulo.pack(pady=(22, 4))

        subtitulo = tk.Label(
            dialogo,
            text="Selecciona un programa existente o crea uno nuevo.",
            font=("Arial", 10),
            fg="#AEB4D0",
            bg=self.color_fondo,
        )
        subtitulo.pack(pady=(0, 16))

        frame_sel = tk.Frame(
            dialogo,
            bg=self.color_panel,
            highlightthickness=1,
            highlightbackground=self.color_borde,
        )
        frame_sel.pack(fill="x", padx=28, pady=7)

        tk.Label(
            frame_sel,
            text="Programa:",
            font=("Arial", 12, "bold"),
            fg="white",
            bg=self.color_panel,
        ).pack(anchor="w", padx=18, pady=(14, 6))

        programas = self.listar_programas()
        variable_programa = tk.StringVar(
            value=programas[0] if programas else "-- Sin programas --"
        )

        combo_programas = ttk.Combobox(
            frame_sel,
            textvariable=variable_programa,
            values=programas or ["-- Sin programas --"],
            state="readonly",
        )
        combo_programas.pack(fill="x", padx=18, pady=(0, 10))

        def cerrar_selector():
            print("[INICIO] Selector cerrado sin abrir programa.")
            seleccionado["ok"] = False
            try:
                dialogo.grab_release()
            except tk.TclError:
                pass
            try:
                dialogo.destroy()
            except tk.TclError:
                pass

        def abrir_programa():
            valor = variable_programa.get().strip()
            if valor not in self.listar_programas():
                return

            print(f"[INICIO] Abriendo programa: {valor}")
            self.programa_actual = valor
            seleccionado["ok"] = True

            try:
                dialogo.grab_release()
            except tk.TclError:
                pass
            dialogo.destroy()

        tk.Button(
            frame_sel,
            text="ABRIR PROGRAMA",
            height=2,
            bg="#1F6AA5",
            fg="white",
            activebackground="#144870",
            activeforeground="white",
            relief="flat",
            command=abrir_programa,
        ).pack(fill="x", padx=18, pady=(0, 6))

        def actualizar_lista_programas():
            nuevos = self.listar_programas()
            combo_programas.configure(
                values=nuevos or ["-- Sin programas --"]
            )
            variable_programa.set(
                nuevos[0] if nuevos else "-- Sin programas --"
            )

        def eliminar_programa():
            valor = variable_programa.get().strip()
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
                    "Error",
                    f"No fue posible eliminar el programa.\n{error}",
                    parent=dialogo,
                )
                return

            actualizar_lista_programas()

        tk.Button(
            frame_sel,
            text="ELIMINAR PROGRAMA",
            height=1,
            bg="#8B2E2E",
            fg="white",
            activebackground="#6E2424",
            activeforeground="white",
            relief="flat",
            command=eliminar_programa,
        ).pack(fill="x", padx=18, pady=(0, 14))

        frame_nuevo = tk.Frame(
            dialogo,
            bg=self.color_panel,
            highlightthickness=1,
            highlightbackground=self.color_borde,
        )
        frame_nuevo.pack(fill="x", padx=28, pady=7)

        tk.Label(
            frame_nuevo,
            text="Crear nuevo programa",
            font=("Arial", 12, "bold"),
            fg="white",
            bg=self.color_panel,
        ).grid(
            row=0, column=0, columnspan=2,
            sticky="w", padx=18, pady=(14, 8)
        )

        entry_nombre_programa = tk.Entry(
            frame_nuevo,
            font=("Arial", 11),
        )
        entry_nombre_programa.grid(
            row=1, column=0,
            padx=(18, 6), pady=6, sticky="ew"
        )

        entry_cantidad_rois = tk.Entry(
            frame_nuevo,
            width=10,
            font=("Arial", 11),
        )
        entry_cantidad_rois.grid(
            row=1, column=1,
            padx=(6, 18), pady=6
        )

        frame_nuevo.grid_columnconfigure(0, weight=1)

        label_estado = tk.Label(
            frame_nuevo,
            text="",
            font=("Arial", 9),
            fg="#FFB86C",
            bg=self.color_panel,
        )
        label_estado.grid(
            row=2, column=0, columnspan=2,
            padx=18, pady=2
        )

        def crear_y_abrir():
            nombre = entry_nombre_programa.get()
            cantidad = entry_cantidad_rois.get()

            if not self.solicitar_clave_admin(dialogo, "crear un programa nuevo"):
                return

            ok, mensaje = self.crear_programa(nombre, cantidad)
            label_estado.configure(
                text=mensaje,
                fg="#6FE3A1" if ok else "#FF6B6B",
            )

            if ok:
                self.programa_actual = nombre.strip()
                seleccionado["ok"] = True
                print(
                    f"[INICIO] Programa creado y abierto: {self.programa_actual}")
                try:
                    dialogo.grab_release()
                except tk.TclError:
                    pass
                dialogo.destroy()

        tk.Button(
            frame_nuevo,
            text="CREAR Y ABRIR",
            height=2,
            bg="#218C4A",
            fg="white",
            activebackground="#176A38",
            activeforeground="white",
            relief="flat",
            command=crear_y_abrir,
        ).grid(
            row=3, column=0, columnspan=2,
            padx=18, pady=(6, 14), sticky="ew"
        )

        dialogo.protocol("WM_DELETE_WINDOW", cerrar_selector)

        try:
            print("[INICIO] Esperando selección de programa...")
            dialogo.grab_set()
            self.ventana.wait_window(dialogo)
        except tk.TclError as error:
            print(f"[INICIO] Error en selector: {error}")
            seleccionado["ok"] = False

        if seleccionado["ok"]:
            try:
                self.ventana.deiconify()
                self.ventana.update_idletasks()
            except tk.TclError:
                return False

        return seleccionado["ok"]

    def cargar_programa(self, nombre_programa):
        """Carga la receta del programa seleccionado desde su programa.ini."""
        self.programa_actual = nombre_programa
        carpeta = os.path.join(self.carpeta_programas, nombre_programa)
        self.ruta_programa = os.path.join(carpeta, "programa.ini")
        self.carpeta_master = os.path.join(carpeta, "masters")
        self.carpeta_resultados = os.path.join(carpeta, "resultados")
        os.makedirs(self.carpeta_master, exist_ok=True)
        os.makedirs(self.carpeta_resultados, exist_ok=True)

        print(f"[PROGRAMA] Archivo utilizado: {self.ruta_programa}")

        config = configparser.ConfigParser()
        try:
            config.read(self.ruta_programa, encoding="utf-8")
        except (configparser.Error, OSError) as error:
            print(f"[PROGRAMA] Error al leer programa.ini: {error}")

        try:
            self.total_rois = config.getint(
                "PROGRAMA", "cantidad_rois", fallback=1)
        except ValueError:
            self.total_rois = 1
        self.total_rois = max(1, min(50, self.total_rois))
        self.rois = self.crear_rois(self.total_rois)
        self.roi_seleccionado = 0
        self.area_pcb = None
        if config.has_section("ALINEACION"):
            try:
                self.area_pcb = tuple(config.getint("ALINEACION", k) for k in ("x1", "y1", "x2", "y2"))
                self.max_desplazamiento = config.getfloat("ALINEACION", "max_desplazamiento", fallback=15.0)
                self.max_rotacion = config.getfloat("ALINEACION", "max_rotacion", fallback=3.0)
                self.min_confianza_alineacion = config.getfloat("ALINEACION", "min_confianza", fallback=0.80)
            except (ValueError, configparser.Error):
                self.area_pcb = None

        for indice, roi in enumerate(self.rois, start=1):
            seccion = f"ROI_{indice}"
            if not config.has_section(seccion):
                continue
            try:
                roi["nombre"] = config.get(
                    seccion, "nombre", fallback=roi["nombre"])
                roi["tipo"] = config.get(
                    seccion, "tipo", fallback=roi["tipo"]
                ).strip().upper()
                if roi["tipo"] not in ("INSPECCION", "DATAMATRIX", "SOLDER_BRIDGE"):
                    roi["tipo"] = "INSPECCION"
                pads_texto = config.get(seccion, "pads", fallback="")
                try:
                    pads = [tuple(map(int, par.split(","))) for par in pads_texto.split(";") if par]
                    roi["pads"] = pads if len(pads) == 4 else []
                except ValueError:
                    roi["pads"] = []
                roi["sat_max"] = config.getint(seccion, "sat_max", fallback=75)
                roi["val_min"] = config.getint(seccion, "val_min", fallback=137)
                roi["dm_longitud"] = config.getint(seccion, "dm_longitud", fallback=16)
                roi["dm_numero_parte"] = config.get(seccion, "dm_numero_parte", fallback="").strip()
                roi["x1"] = config.getint(seccion, "x1", fallback=roi["x1"])
                roi["y1"] = config.getint(seccion, "y1", fallback=roi["y1"])
                roi["x2"] = config.getint(seccion, "x2", fallback=roi["x2"])
                roi["y2"] = config.getint(seccion, "y2", fallback=roi["y2"])
                roi["similitud_minima"] = config.getfloat(
                    seccion, "similitud_minima", fallback=roi["similitud_minima"]
                )
                roi["sensibilidad"] = config.getint(
                    seccion, "sensibilidad", fallback=roi["sensibilidad"]
                )
                roi["area_minima"] = config.getint(
                    seccion, "area_minima", fallback=roi["area_minima"]
                )
                roi["area_defecto_maxima"] = config.getfloat(
                    seccion, "area_defecto_maxima", fallback=roi["area_defecto_maxima"]
                )
                roi["porcentaje_diferente_maximo"] = config.getfloat(
                    seccion,
                    "porcentaje_diferente_maximo",
                    fallback=roi["porcentaje_diferente_maximo"],
                )
            except ValueError as error:
                print(f"[PROGRAMA] Valor no válido en {seccion}: {error}")

        print(
            f"[PROGRAMA] '{self.programa_actual}' cargado con {self.total_rois} ROI(s).")
        for indice, roi in enumerate(self.rois, start=1):
            print(
                f"[PROGRAMA] ROI_{indice}: "
                f"similitud={roi['similitud_minima']}, "
                f"sensibilidad={roi['sensibilidad']}, "
                f"area_minima={roi['area_minima']}, "
                f"area_defecto_maxima={roi['area_defecto_maxima']}, "
                f"porcentaje_diferente_maximo={roi['porcentaje_diferente_maximo']}"
            )

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

        # Panel derecho dividido en tres zonas:
        # 1) estado fijo arriba, 2) controles desplazables, 3) INICIAR PRUEBA fijo abajo.
        panel_derecho = ctk.CTkFrame(
            contenido,
            width=350,
            fg_color=self.color_panel,
            border_width=1,
            border_color=self.color_borde,
            corner_radius=12,
        )
        panel_derecho.grid(row=0, column=1, sticky="ns")
        panel_derecho.grid_columnconfigure(0, weight=1)
        panel_derecho.grid_rowconfigure(1, weight=1)

        self.label_resultado_general = ctk.CTkLabel(
            panel_derecho,
            text="En espera",
            height=60,
            corner_radius=10,
            fg_color="#555555",
            font=("Arial", 23, "bold"),
            text_color="white",
        )
        self.label_resultado_general.grid(
            row=0, column=0, padx=16, pady=(12, 6), sticky="ew")

        panel = ctk.CTkScrollableFrame(
            panel_derecho,
            width=318,
            fg_color="transparent",
            corner_radius=0,
        )
        panel.grid(row=1, column=0, padx=(6, 4), pady=0, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)

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
        ).grid(row=3, column=0, padx=18, pady=(8, 2), sticky="w")

        self.menu_roi = ctk.CTkOptionMenu(
            panel,
            values=[f"ROI {i}" for i in range(1, self.total_rois + 1)],
            command=self.cambiar_roi_seleccionado,
        )
        self.menu_roi.grid(row=4, column=0, padx=18, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(
            panel, text="Tipo de ROI", font=("Arial", 12, "bold"),
            text_color="#AEB4D0"
        ).grid(row=5, column=0, padx=18, pady=(4, 2), sticky="w")

        self.menu_tipo_roi = ctk.CTkOptionMenu(
            panel, values=["INSPECCION", "DATAMATRIX", "SOLDER_BRIDGE"],
            command=self.cambiar_tipo_roi
        )
        self.menu_tipo_roi.grid(row=6, column=0, padx=18,
                                pady=(0, 8), sticky="ew")
        self.boton_configurar_dm = ctk.CTkButton(
            panel, text="CONFIGURAR DATAMATRIX (ID / NÚMERO DE PARTE)",
            command=self.configurar_datamatrix)
        self.boton_configurar_dm.grid(row=35, column=0, padx=18, pady=(6, 8), sticky="ew")
        self.boton_configurar_dm.grid_remove()

        self.label_coordenadas = ctk.CTkLabel(
            panel,
            text="X1: ---  Y1: ---  X2: ---  Y2: ---",
            font=("Arial", 12),
            text_color="#AEB4D0",
        )
        self.label_coordenadas.grid(
            row=7, column=0, padx=18, pady=4, sticky="ew")

        self.label_alineacion = ctk.CTkLabel(
            panel, text="SIN REFERENCIA PCB", text_color="#F5C84B",
            font=("Arial", 11, "bold"), wraplength=285)
        self.label_alineacion.grid(row=29, column=0, padx=18, pady=5, sticky="ew")
        self.boton_area_pcb = ctk.CTkButton(
            panel, text="CONFIGURAR ÁREA PCB", command=self.configurar_area_pcb)
        self.boton_area_pcb.grid(row=30, column=0, padx=18, pady=5, sticky="ew")
        self.boton_referencia_pcb = ctk.CTkButton(
            panel, text="CAPTURAR REFERENCIA PCB", command=self.capturar_referencia_pcb)
        self.boton_referencia_pcb.grid(row=31, column=0, padx=18, pady=5, sticky="ew")
        self.label_masters = ctk.CTkLabel(panel, text="Masters ROI: 0/5", text_color="#AEB4D0")
        self.label_masters.grid(row=32, column=0, padx=18, pady=5, sticky="ew")
        self.boton_borrar_masters = ctk.CTkButton(
            panel, text="REINICIAR MASTERS DEL ROI", fg_color="#7A3535",
            command=self.reiniciar_masters_roi)
        self.boton_borrar_masters.grid(row=33, column=0, padx=18, pady=5, sticky="ew")

        self.boton_admin = ctk.CTkButton(
            panel,
            text="MODO ADMINISTRADOR",
            height=42,
            font=("Arial", 13, "bold"),
            fg_color="#7A5C00",
            hover_color="#5F4700",
            command=self.alternar_modo_admin,
        )
        self.boton_admin.grid(row=8, column=0, padx=18,
                              pady=(8, 6), sticky="ew")

        self.boton_configurar = ctk.CTkButton(
            panel,
            text="CONFIGURAR ÁREA CON MOUSE",
            height=42,
            font=("Arial", 13, "bold"),
            command=self.activar_configuracion_roi,
        )
        self.boton_configurar.grid(
            row=9, column=0, padx=18, pady=6, sticky="ew")

        self.boton_master = ctk.CTkButton(
            panel,
            text="AGREGAR MASTER DEL ROI (1-5)",
            height=42,
            font=("Arial", 13, "bold"),
            fg_color="#315A77",
            hover_color="#26465D",
            command=self.capturar_master_roi,
        )
        self.boton_master.grid(row=10, column=0, padx=18, pady=6, sticky="ew")

        self.boton_todos_masters = ctk.CTkButton(
            panel,
            text="AGREGAR MASTER A TODOS LOS ROIs",
            height=42,
            font=("Arial", 13, "bold"),
            fg_color="#6C4F8A",
            hover_color="#533D6B",
            command=self.capturar_todos_los_masters,
        )
        self.boton_todos_masters.grid(
            row=11, column=0, padx=18, pady=6, sticky="ew")

        ctk.CTkLabel(
            panel,
            text="Similitud mínima",
            font=("Arial", 14, "bold"),
            text_color="#AEB4D0",
        ).grid(row=12, column=0, padx=18, pady=(16, 3), sticky="w")

        self.label_similitud_config = ctk.CTkLabel(
            panel,
            text="90.0 %",
            font=("Arial", 16, "bold"),
            text_color="white",
        )
        self.label_similitud_config.grid(row=13, column=0, padx=18, pady=2)

        self.slider_similitud = ctk.CTkSlider(
            panel,
            from_=50,
            to=100,
            number_of_steps=100,
            command=self.cambiar_similitud_roi,
        )
        self.slider_similitud.grid(
            row=14, column=0, padx=18, pady=(2, 8), sticky="ew")

        ctk.CTkLabel(
            panel,
            text="Sensibilidad de diferencias",
            font=("Arial", 14, "bold"),
            text_color="#AEB4D0",
        ).grid(row=15, column=0, padx=18, pady=(10, 3), sticky="w")

        self.label_sensibilidad = ctk.CTkLabel(
            panel,
            text="70 %",
            font=("Arial", 16, "bold"),
            text_color="white",
        )
        self.label_sensibilidad.grid(row=16, column=0, padx=18, pady=2)

        self.slider_sensibilidad = ctk.CTkSlider(
            panel,
            from_=1,
            to=100,
            number_of_steps=99,
            command=self.cambiar_sensibilidad_roi,
        )
        self.slider_sensibilidad.grid(
            row=17, column=0, padx=18, pady=(2, 4), sticky="ew")

        self.label_ayuda_sensibilidad = ctk.CTkLabel(
            panel,
            text="Mayor sensibilidad = detecta diferencias más pequeñas.",
            font=("Arial", 11),
            text_color="#AEB4D0",
            wraplength=280,
        )
        self.label_ayuda_sensibilidad.grid(
            row=18, column=0, padx=18, pady=(0, 10))

        # Parámetros avanzados por ROI. Se editan sólo en modo administrador.
        ctk.CTkLabel(panel, text="Área mínima de defecto (px²)", font=("Arial", 12, "bold"),
                     text_color="#AEB4D0").grid(row=19, column=0, padx=18, pady=(8, 2), sticky="w")
        self.label_area_minima = ctk.CTkLabel(
            panel, text="20 px²", text_color="white")
        self.label_area_minima.grid(row=20, column=0, padx=18, pady=1)
        self.slider_area_minima = ctk.CTkSlider(panel, from_=1, to=500, number_of_steps=499,
                                                command=self.cambiar_area_minima_roi)
        self.slider_area_minima.grid(
            row=21, column=0, padx=18, pady=(1, 6), sticky="ew")

        ctk.CTkLabel(panel, text="Área máxima permitida (px²)", font=("Arial", 12, "bold"),
                     text_color="#AEB4D0").grid(row=22, column=0, padx=18, pady=(8, 2), sticky="w")
        self.label_area_maxima = ctk.CTkLabel(
            panel, text="50 px²", text_color="white")
        self.label_area_maxima.grid(row=23, column=0, padx=18, pady=1)
        self.slider_area_maxima = ctk.CTkSlider(panel, from_=1, to=5000, number_of_steps=999,
                                                command=self.cambiar_area_maxima_roi)
        self.slider_area_maxima.grid(
            row=24, column=0, padx=18, pady=(1, 6), sticky="ew")

        ctk.CTkLabel(panel, text="Diferencia máxima permitida (%)", font=("Arial", 12, "bold"),
                     text_color="#AEB4D0").grid(row=25, column=0, padx=18, pady=(8, 2), sticky="w")
        self.label_porcentaje_diferente = ctk.CTkLabel(
            panel, text="0.200 %", text_color="white")
        self.label_porcentaje_diferente.grid(row=26, column=0, padx=18, pady=1)
        self.slider_porcentaje_diferente = ctk.CTkSlider(panel, from_=0.01, to=10.0, number_of_steps=999,
                                                         command=self.cambiar_porcentaje_diferente_roi)
        self.slider_porcentaje_diferente.grid(
            row=27, column=0, padx=18, pady=(1, 8), sticky="ew")

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
            row=28, column=0, padx=18, pady=(12, 6), sticky="ew")

        self.boton_calibrar_pads = ctk.CTkButton(
            panel, text="CALIBRAR 4 PADS (SOLDER BRIDGE)", height=42,
            fg_color="#6B557A", command=self.activar_calibracion_pads,
        )
        self.boton_calibrar_pads.grid(row=29, column=0, padx=18, pady=(6, 12), sticky="ew")

        # Rev7: los controles HSV viven en la ventana de calibración.
        self.boton_mascara = ctk.CTkButton(
            panel, text="CALIBRAR MÁSCARA DE DETECCIÓN",
            fg_color="#31647B", command=self.ver_mascara_roi)
        self.boton_mascara.grid(row=34, column=0, padx=18, pady=(8, 14), sticky="ew")

        self.boton_guardar = ctk.CTkButton(
            panel_derecho,
            text="INICIAR PRUEBA",
            height=72,
            font=("Arial", 26, "bold"),
            fg_color="#8A6D1D",
            hover_color="#6C5515",
            command=self.guardar_resultado,
        )
        self.boton_guardar.grid(
            row=2, column=0, padx=16, pady=(6, 12), sticky="ew")

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
            self.boton_configurar_dm,
            self.boton_calibrar_pads,
            self.boton_area_pcb,
            self.boton_referencia_pcb,
            self.boton_borrar_masters,
            self.boton_master,
            self.boton_todos_masters,
            self.slider_similitud,
            self.slider_sensibilidad,
            self.boton_mascara,
            self.slider_area_minima,
            self.slider_area_maxima,
            self.slider_porcentaje_diferente,
            self.menu_tipo_roi,
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
        """Carga config.ini desde la misma carpeta del script."""
        config = configparser.ConfigParser()

        print(f"[CONFIG] Archivo utilizado: {self.ruta_configuracion}")

        if os.path.exists(self.ruta_configuracion):
            try:
                archivos_leidos = config.read(
                    self.ruta_configuracion, encoding="utf-8")
                if not archivos_leidos:
                    print("[CONFIG] Advertencia: config.ini no pudo ser leído.")
            except (configparser.Error, OSError) as error:
                print(f"[CONFIG] No fue posible cargar config.ini: {error}")
                config = configparser.ConfigParser()
        else:
            print(
                "[CONFIG] config.ini no existe. Se creará con valores predeterminados.")

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
                print(f"[CONFIG] Configuración de cámara no válida: {error}")

        if config.has_section("ADMIN"):
            clave = config.get(
                "ADMIN", "clave", fallback=self.clave_admin).strip()
            if clave:
                self.clave_admin = clave

        print(
            "[CONFIG] Cámara cargada -> "
            f"indice={self.indice_camara}, ancho={self.ancho_camara}, alto={self.alto_camara}"
        )
        print("[CONFIG] Sección ADMIN cargada correctamente.")

        # Sólo completa secciones/valores faltantes. No reemplaza valores válidos
        # que el usuario haya modificado manualmente.
        modificado = False

        if not config.has_section("CAMARA"):
            config.add_section("CAMARA")
            modificado = True

        valores_camara = {
            "indice": str(self.indice_camara),
            "ancho": str(self.ancho_camara),
            "alto": str(self.alto_camara),
        }
        for clave, valor in valores_camara.items():
            if not config.has_option("CAMARA", clave):
                config.set("CAMARA", clave, valor)
                modificado = True

        if not config.has_section("ADMIN"):
            config.add_section("ADMIN")
            modificado = True
        if not config.has_option("ADMIN", "clave") or not config.get("ADMIN", "clave").strip():
            config.set("ADMIN", "clave", self.clave_admin)
            modificado = True

        if modificado:
            try:
                with open(self.ruta_configuracion, "w", encoding="utf-8") as archivo:
                    config.write(archivo)
            except OSError as error:
                print(
                    f"[CONFIG] No fue posible actualizar config.ini: {error}")

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
                "tipo": roi["tipo"],
                "pads": ";".join(f"{x},{y}" for x, y in roi.get("pads", [])),
                "sat_max": str(int(roi.get("sat_max", 75))),
                "val_min": str(int(roi.get("val_min", 137))),
                "dm_longitud": str(int(roi.get("dm_longitud", 16))),
                "dm_numero_parte": str(roi.get("dm_numero_parte", "")),
                "x1": str(roi["x1"]), "y1": str(roi["y1"]),
                "x2": str(roi["x2"]), "y2": str(roi["y2"]),
                "similitud_minima": f'{roi["similitud_minima"]:.2f}',
                "sensibilidad": str(int(roi["sensibilidad"])),
                "area_minima": str(int(roi["area_minima"])),
                "area_defecto_maxima": f'{roi["area_defecto_maxima"]:.2f}',
                "porcentaje_diferente_maximo": f'{roi["porcentaje_diferente_maximo"]:.4f}',
            }
        if self.area_pcb is not None:
            x1, y1, x2, y2 = self.area_pcb
            config["ALINEACION"] = {
                "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2),
                "max_desplazamiento": str(self.max_desplazamiento),
                "max_rotacion": str(self.max_rotacion),
                "min_confianza": str(self.min_confianza_alineacion),
            }
        with open(self.ruta_programa, "w", encoding="utf-8") as archivo:
            config.write(archivo)

    def ruta_master_roi(self, indice, numero=1):
        return os.path.join(self.carpeta_master, f"roi_{indice + 1}_{numero}.png")

    def ruta_referencia_pcb(self):
        return os.path.join(self.carpeta_master, "referencia_pcb.png")

    def cargar_referencia_pcb(self):
        ruta = self.ruta_referencia_pcb()
        self.referencia_pcb = cv2.imread(ruta) if os.path.isfile(ruta) else None
        self.alineacion_info = "REFERENCIA LISTA" if self.referencia_pcb is not None else "SIN REFERENCIA PCB"
        if hasattr(self, "label_alineacion"):
            self.label_alineacion.configure(text=self.alineacion_info)

    def cargar_masters(self):
        for indice, roi in enumerate(self.rois):
            imagenes = []
            for numero in range(1, self.max_masters + 1):
                ruta = self.ruta_master_roi(indice, numero)
                imagen = cv2.imread(ruta) if os.path.isfile(ruta) else None
                if imagen is not None:
                    imagenes.append(imagen)
            # Compatibilidad con la versión anterior, sin eliminar su master.
            ruta_legacy = os.path.join(self.carpeta_master, f"roi_{indice + 1}.jpg")
            if not imagenes and os.path.isfile(ruta_legacy):
                imagen = cv2.imread(ruta_legacy)
                if imagen is not None:
                    imagenes.append(imagen)
            roi["masters"] = imagenes
            roi["master"] = imagenes[0] if imagenes else None
            roi["resultado"] = "LISTO" if imagenes else "SIN MASTER"

    def obtener_frame_alineado(self, frame):
        """ECC: transforma la fotografía actual al sistema de coordenadas de referencia.

        Retorna (frame_alineado, mensaje). Si no hay referencia, no permite PASS.
        """
        if self.referencia_pcb is None or self.area_pcb is None:
            return None, "SIN REFERENCIA PCB / ÁREA"
        referencia = self.referencia_pcb
        if frame.shape[:2] != referencia.shape[:2]:
            return None, "RESOLUCIÓN DE CÁMARA CAMBIÓ"
        x1, y1, x2, y2 = self.area_pcb
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 50 or y2 - y1 < 50:
            return None, "ÁREA PCB INVÁLIDA"
        gris_ref = self.preparar_imagen(referencia[y1:y2, x1:x2])
        gris_actual = self.preparar_imagen(frame[y1:y2, x1:x2])
        # ECC local: se espera que el fixture limite el desplazamiento.
        matriz = np.eye(2, 3, dtype=np.float32)
        criterios = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
        try:
            confianza, matriz = cv2.findTransformECC(
                gris_ref, gris_actual, matriz, cv2.MOTION_EUCLIDEAN,
                criterios, None, 5
            )
        except cv2.error:
            return None, "NO SE PUDO ALINEAR PCB"
        angulo = float(np.degrees(np.arctan2(matriz[1, 0], matriz[0, 0])))
        dx, dy = float(matriz[0, 2]), float(matriz[1, 2])
        if (confianza < self.min_confianza_alineacion or
                abs(dx) > self.max_desplazamiento or
                abs(dy) > self.max_desplazamiento or
                abs(angulo) > self.max_rotacion):
            return None, f"ALINEACIÓN FUERA DE LÍMITE (X:{dx:.1f} Y:{dy:.1f} R:{angulo:.2f} C:{confianza:.2f})"
        # La matriz ECC está en coordenadas del recorte; convertir a imagen completa.
        rotacion = matriz[:, :2].astype(np.float64)
        origen = np.array([x1, y1], dtype=np.float64)
        matriz_global = matriz.copy()
        matriz_global[:, 2] = (matriz[:, 2] + origen - rotacion @ origen)
        alineado = cv2.warpAffine(frame, matriz_global, (w, h),
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
        return alineado, f"X:{dx:+.1f}px Y:{dy:+.1f}px R:{angulo:+.2f}° C:{confianza:.3f}"

    def capturar_referencia_pcb(self):
        if not self.modo_admin or self.frame_actual is None or self.area_pcb is None:
            messagebox.showwarning("Referencia PCB", "Activa administrador y define primero el área PCB.")
            return
        if not messagebox.askyesno("Referencia PCB", "¿Guardar la PCB buena actual como referencia de alineación?\nEsto invalidará la calibración anterior."):
            return
        if not cv2.imwrite(self.ruta_referencia_pcb(), self.frame_actual):
            messagebox.showerror("Error", "No se pudo guardar la referencia PCB")
            return
        self.referencia_pcb = self.frame_actual.copy()
        self.alineacion_info = "REFERENCIA PCB GUARDADA"
        self.label_alineacion.configure(text=self.alineacion_info)
        self.label_resultado_general.configure(text="REFERENCIA PCB GUARDADA", fg_color="#315A77")

    def configurar_area_pcb(self):
        if not self.modo_admin:
            return
        self.configurando_roi = False
        self.configurando_area_pcb = True
        self.roi_arrastrando = False
        self.label_resultado_general.configure(text="DIBUJA ÁREA PCB", fg_color="#A66A00")

    def iniciar_camara(self):
        """Abre la cámara usando el backend adecuado para Windows o Linux/Ubuntu."""
        sistema = platform.system()

        if sistema == "Windows":
            backend = cv2.CAP_DSHOW
            nombre_backend = "DirectShow"
        elif sistema == "Linux":
            backend = cv2.CAP_V4L2
            nombre_backend = "V4L2"
        else:
            # Deja que OpenCV seleccione automáticamente el backend en otros SO.
            backend = cv2.CAP_ANY
            nombre_backend = "automático"

        print(
            f"[CAMARA] Sistema={sistema} | backend={nombre_backend} | indice={self.indice_camara}")
        self.camara = cv2.VideoCapture(self.indice_camara, backend)

        # Si el backend preferido falla, intentar selección automática.
        if not self.camara.isOpened() and backend != cv2.CAP_ANY:
            print(
                f"[CAMARA] No abrió con {nombre_backend}. Intentando backend automático...")
            self.camara.release()
            self.camara = cv2.VideoCapture(self.indice_camara, cv2.CAP_ANY)

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
        self.menu_tipo_roi.set(roi.get("tipo", "INSPECCION"))
        if hasattr(self, "boton_configurar_dm"):
            if roi.get("tipo") == "DATAMATRIX":
                self.boton_configurar_dm.grid()
            else:
                self.boton_configurar_dm.grid_remove()
        self.slider_similitud.set(roi["similitud_minima"])
        self.slider_sensibilidad.set(roi["sensibilidad"])
        self.slider_area_minima.set(roi["area_minima"])
        self.slider_area_maxima.set(roi["area_defecto_maxima"])
        self.slider_porcentaje_diferente.set(
            roi["porcentaje_diferente_maximo"])
        self.label_area_minima.configure(text=f'{int(roi["area_minima"])} px²')
        self.label_area_maxima.configure(
            text=f'{roi["area_defecto_maxima"]:.0f} px²')
        self.label_porcentaje_diferente.configure(
            text=f'{roi["porcentaje_diferente_maximo"]:.3f} %')
        self.label_similitud_config.configure(
            text=f'{roi["similitud_minima"]:.1f} %')
        self.label_sensibilidad.configure(text=f'{int(roi["sensibilidad"])} %')
        if hasattr(self, "label_masters"):
            self.label_masters.configure(text=f'Masters ROI: {len(roi.get("masters", []))}/{self.max_masters}')
        self.label_coordenadas.configure(
            text=(
                f'X1: {roi["x1"]}  Y1: {roi["y1"]}  '
                f'X2: {roi["x2"]}  Y2: {roi["y2"]}'
            )
        )

    def cambiar_tipo_roi(self, valor):
        if not self.modo_admin:
            return
        roi = self.rois[self.roi_seleccionado]
        roi["tipo"] = str(valor).strip().upper()
        if roi["tipo"] == "SOLDER_BRIDGE" and len(roi.get("pads", [])) != 4:
            roi["resultado"] = "CALIBRAR 4 PADS"
        if roi["tipo"] == "DATAMATRIX":
            roi["master"] = None
            roi["resultado"] = "LISTO DATAMATRIX"
            roi["codigo"] = ""
        self.guardar_configuracion()
        self.actualizar_controles_roi()
        self.actualizar_etiquetas_resultados()

    def configurar_datamatrix(self):
        """Configura el largo exacto y el prefijo del ID para el ROI seleccionado."""
        if not self.modo_admin:
            return
        roi = self.rois[self.roi_seleccionado]
        if roi.get("tipo") != "DATAMATRIX":
            return
        ventana = ctk.CTkToplevel(self.ventana)
        ventana.title(f"Configurar Data Matrix - ROI {self.roi_seleccionado + 1}")
        ventana.geometry("490x350")
        ventana.resizable(False, False)
        ventana.transient(self.ventana)
        ventana.lift()
        ventana.grab_set()
        ventana.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ventana, text="Longitud exacta del ID (caracteres)").grid(
            row=0, column=0, padx=22, pady=(20, 4), sticky="w")
        entrada_longitud = ctk.CTkEntry(ventana)
        entrada_longitud.insert(0, str(roi.get("dm_longitud", 16)))
        entrada_longitud.grid(row=1, column=0, padx=22, sticky="ew")
        ctk.CTkLabel(ventana, text="Número de parte esperado (prefijo del ID)").grid(
            row=2, column=0, padx=22, pady=(14, 4), sticky="w")
        entrada_parte = ctk.CTkEntry(ventana)
        entrada_parte.insert(0, roi.get("dm_numero_parte", ""))
        entrada_parte.grid(row=3, column=0, padx=22, sticky="ew")
        ctk.CTkLabel(ventana,
            text="El ID debe tener la longitud indicada y comenzar exactamente\n"
                 "con el número de parte. El ID debe ser más largo que el prefijo.",
            text_color="#AEB4D0", justify="left").grid(
                row=4, column=0, padx=22, pady=14, sticky="w")
        def guardar():
            try:
                longitud = int(entrada_longitud.get().strip())
            except ValueError:
                messagebox.showerror("Data Matrix", "La longitud debe ser un número entero.", parent=ventana)
                return
            parte = entrada_parte.get().strip()
            if not parte or longitud <= len(parte) or longitud > 512:
                messagebox.showerror("Data Matrix",
                    "Indica un número de parte y una longitud mayor que la del número de parte (máximo 512).",
                    parent=ventana)
                return
            if any(c in parte for c in "\r\n\t"):
                messagebox.showerror("Data Matrix", "Número de parte no válido.", parent=ventana)
                return
            roi["dm_longitud"] = longitud
            roi["dm_numero_parte"] = parte
            self.guardar_configuracion()
            ventana.grab_release()
            ventana.destroy()
            self.label_resultado_general.configure(text="DATAMATRIX CONFIGURADO", fg_color="#315A77")
        ctk.CTkButton(ventana, text="GUARDAR CONFIGURACIÓN", command=guardar).grid(
            row=5, column=0, padx=22, pady=6, sticky="ew")
        ctk.CTkButton(ventana, text="CANCELAR", fg_color="#555555",
            command=ventana.destroy).grid(row=6, column=0, padx=22, pady=4, sticky="ew")

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

    def cambiar_area_minima_roi(self, valor):
        if not self.modo_admin:
            return
        roi = self.rois[self.roi_seleccionado]
        roi["area_minima"] = max(1, int(round(float(valor))))
        self.label_area_minima.configure(text=f'{roi["area_minima"]} px²')
        self.guardar_configuracion()

    def cambiar_area_maxima_roi(self, valor):
        if not self.modo_admin:
            return
        roi = self.rois[self.roi_seleccionado]
        roi["area_defecto_maxima"] = max(1.0, float(valor))
        self.label_area_maxima.configure(
            text=f'{roi["area_defecto_maxima"]:.0f} px²')
        self.guardar_configuracion()

    def cambiar_porcentaje_diferente_roi(self, valor):
        if not self.modo_admin:
            return
        roi = self.rois[self.roi_seleccionado]
        roi["porcentaje_diferente_maximo"] = max(0.01, float(valor))
        self.label_porcentaje_diferente.configure(
            text=f'{roi["porcentaje_diferente_maximo"]:.3f} %')
        self.guardar_configuracion()

    def activar_configuracion_roi(self):

        if not self.modo_admin:
            return
        self.comparacion_activa = False
        self.calibrando_pads = False
        self.configurando_roi = True
        self.configurando_area_pcb = False
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

    def activar_calibracion_pads(self):
        """El administrador selecciona el centro de cada uno de los 4 pads."""
        if not self.modo_admin:
            return
        roi = self.rois[self.roi_seleccionado]
        if roi.get("tipo") != "SOLDER_BRIDGE":
            messagebox.showwarning("Tipo de ROI", "Selecciona primero SOLDER_BRIDGE.")
            return
        self.comparacion_activa = False
        self.mostrando_fotografia = False
        self.configurando_roi = False
        self.calibrando_pads = True
        self.pads_temporales = []
        self.label_resultado_general.configure(text="SELECCIONA PAD 1 DE 4", fg_color="#6B557A")

    def seleccionar_pad_mouse(self, evento):
        if not self.calibrando_pads:
            return False
        punto = self.coordenada_pantalla_a_frame(evento.x, evento.y)
        if punto is None:
            return True
        roi = self.rois[self.roi_seleccionado]
        x, y = punto
        if not (roi["x1"] <= x < roi["x2"] and roi["y1"] <= y < roi["y2"]):
            self.label_resultado_general.configure(text="PAD FUERA DEL ROI", fg_color="#B3261E")
            return True
        local = (x-roi["x1"], y-roi["y1"])
        if any(np.hypot(x-px, y-py) < 5 for px, py in [(a+roi["x1"],b+roi["y1"]) for a,b in self.pads_temporales]):
            return True
        self.pads_temporales.append(local)
        if len(self.pads_temporales) == 4:
            roi["pads"] = list(self.pads_temporales)
            self.guardar_configuracion()
            self.calibrando_pads = False
            self.label_resultado_general.configure(text="4 PADS CALIBRADOS", fg_color="#218C4A")
        else:
            self.label_resultado_general.configure(text=f"SELECCIONA PAD {len(self.pads_temporales)+1} DE 4", fg_color="#6B557A")
        return True

    def iniciar_dibujo_roi(self, evento):
        if self.seleccionar_pad_mouse(evento):
            return
        if not (self.configurando_roi or self.configurando_area_pcb):
            return

        punto = self.coordenada_pantalla_a_frame(evento.x, evento.y)
        if punto is None:
            return

        self.roi_arrastrando = True
        self.punto_inicio_roi = punto
        self.roi_temporal = (*punto, *punto)

    def actualizar_dibujo_roi(self, evento):
        if not (self.configurando_roi or self.configurando_area_pcb) or not self.roi_arrastrando:
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
        if not (self.configurando_roi or self.configurando_area_pcb) or not self.roi_arrastrando:
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

        if self.configurando_area_pcb:
            self.area_pcb = (x1, y1, x2, y2)
            self.configurando_area_pcb = False
            self.roi_temporal = None
            self.punto_inicio_roi = None
            self.referencia_pcb = None
            ruta_ref = self.ruta_referencia_pcb()
            if os.path.isfile(ruta_ref):
                os.remove(ruta_ref)
            self.guardar_configuracion()
            self.label_alineacion.configure(text="ÁREA LISTA: CAPTURA REFERENCIA PCB")
            self.label_resultado_general.configure(text="ÁREA PCB GUARDADA", fg_color="#315A77")
            return
        roi = self.rois[self.roi_seleccionado]
        roi.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        # Al cambiar el tamaño del ROI, la imagen MASTER anterior deja de ser válida.
        roi["master"] = None
        roi["masters"] = []
        roi["pads"] = []
        roi["resultado"] = "SIN MASTER"
        for numero in range(1, self.max_masters + 1):
            ruta = self.ruta_master_roi(self.roi_seleccionado, numero)
            if os.path.isfile(ruta):
                os.remove(ruta)
        ruta_legacy = os.path.join(self.carpeta_master, f"roi_{self.roi_seleccionado + 1}.jpg")
        if os.path.isfile(ruta_legacy):
            os.remove(ruta_legacy)

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

    def reiniciar_masters_roi(self):
        if not self.modo_admin:
            return
        indice = self.roi_seleccionado
        if not messagebox.askyesno("Reiniciar masters", f"¿Eliminar los masters del ROI {indice + 1} para volver a capturarlos?"):
            return
        for numero in range(1, self.max_masters + 1):
            ruta = self.ruta_master_roi(indice, numero)
            if os.path.isfile(ruta):
                os.remove(ruta)
        ruta_legacy = os.path.join(self.carpeta_master, f"roi_{indice + 1}.jpg")
        if os.path.isfile(ruta_legacy):
            os.remove(ruta_legacy)
        roi = self.rois[indice]
        roi["masters"] = []
        roi["master"] = None
        roi["resultado"] = "SIN MASTER"
        roi["similitud"] = None
        self.actualizar_controles_roi()
        self.actualizar_etiquetas_resultados()

    def capturar_master_roi(self):
        if not self.modo_admin or self.frame_actual is None:
            return
        roi = self.rois[self.roi_seleccionado]
        if roi.get("tipo") in ("DATAMATRIX", "SOLDER_BRIDGE"):
            self.label_resultado_general.configure(text="ESTE TIPO DE ROI NO REQUIERE MASTER", fg_color="#315A77")
            return
        alineado, motivo = self.obtener_frame_alineado(self.frame_actual)
        if alineado is None:
            self.label_resultado_general.configure(text="ERROR ALINEACIÓN", fg_color="#B3261E")
            self.label_alineacion.configure(text=motivo)
            return
        imagen_roi = self.obtener_roi(alineado, roi)
        if imagen_roi.size == 0:
            return
        numero = len(roi.get("masters", [])) + 1
        if numero > self.max_masters:
            messagebox.showinfo("Masters completos", "Ya existen 5 masters. Para reemplazarlos, elimina los masters del ROI desde la carpeta masters y reinicia el programa.")
            return
        if not cv2.imwrite(self.ruta_master_roi(self.roi_seleccionado, numero), imagen_roi):
            self.label_resultado_general.configure(text="ERROR AL GUARDAR", fg_color="#B3261E")
            return
        roi.setdefault("masters", []).append(imagen_roi.copy())
        roi["master"] = roi["masters"][0]
        roi["resultado"] = "LISTO"
        self.label_resultado_general.configure(text=f"MASTER {numero}/5 GUARDADO", fg_color="#315A77")
        self.label_alineacion.configure(text=motivo)
        self.actualizar_controles_roi()
        self.actualizar_etiquetas_resultados()

    def capturar_todos_los_masters(self):
        if not self.modo_admin or self.frame_actual is None:
            return
        alineado, motivo = self.obtener_frame_alineado(self.frame_actual)
        if alineado is None:
            self.label_alineacion.configure(text=motivo)
            self.label_resultado_general.configure(text="ERROR ALINEACIÓN", fg_color="#B3261E")
            return
        guardados = 0
        for indice, roi in enumerate(self.rois):
            if roi.get("tipo") in ("DATAMATRIX", "SOLDER_BRIDGE") or len(roi.get("masters", [])) >= self.max_masters:
                continue
            imagen = self.obtener_roi(alineado, roi)
            numero = len(roi["masters"]) + 1
            if imagen.size and cv2.imwrite(self.ruta_master_roi(indice, numero), imagen):
                roi["masters"].append(imagen.copy())
                roi["master"] = roi["masters"][0]
                roi["resultado"] = "LISTO"
                guardados += 1
        self.label_resultado_general.configure(text=f"{guardados} MASTERS GUARDADOS", fg_color="#315A77")
        self.actualizar_controles_roi()
        self.actualizar_etiquetas_resultados()

    @staticmethod
    def preparar_imagen(imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gris, (3, 3), 0)

    @staticmethod
    def calcular_umbral_diferencia(sensibilidad):
        """
        Convierte sensibilidad 1-100 a umbral binario.
        Mayor sensibilidad produce un umbral menor y detecta cambios más pequeños.
        """
        sensibilidad = max(1, min(100, int(sensibilidad)))
        # 109 a 30 aproximadamente.
        return int(round(110 - (sensibilidad * 0.8)))

    @staticmethod
    def leer_data_matrix(imagen_roi):
        """Lee el primer Data Matrix encontrado dentro del ROI."""
        if zxingcpp is None or imagen_roi is None or imagen_roi.size == 0:
            return None
        try:
            gris = cv2.cvtColor(imagen_roi, cv2.COLOR_BGR2GRAY)
            resultados = zxingcpp.read_barcodes(
                gris)
            for resultado in resultados:
                if resultado.format == zxingcpp.BarcodeFormat.DataMatrix:
                    return resultado.text.strip()
        except Exception as error:
            print(f"[DATAMATRIX] Error de lectura: {error}")
        return None

    def ver_mascara_roi(self):
        """Calibrador modal, en vivo, con cambios temporales y Guardar/Cancelar."""
        if not self.modo_admin:
            messagebox.showinfo("Calibración", "Activa el modo administrador.")
            return
        indice = self.roi_seleccionado
        roi = self.rois[indice]
        if roi.get("tipo") != "SOLDER_BRIDGE":
            messagebox.showinfo("Máscara", "Selecciona un ROI SOLDER_BRIDGE.")
            return
        if self.frame_actual is None:
            messagebox.showwarning("Máscara", "No hay imagen de cámara disponible.")
            return
        if self.ventana_mascara is not None:
            try:
                if self.ventana_mascara.winfo_exists():
                    self.ventana_mascara.lift()
                    self.ventana_mascara.focus_force()
                    return
            except tk.TclError:
                pass

        ventana = ctk.CTkToplevel(self.ventana)
        self.ventana_mascara = ventana
        ventana.title(f"Calibración en vivo - ROI {indice + 1}")
        ventana.geometry("920x660")
        ventana.minsize(650, 510)
        ventana.transient(self.ventana)
        ventana.lift()
        ventana.grab_set()
        ventana.focus_force()

        estado = {"activo": True, "after": None, "foto": None}
        sat_var = tk.IntVar(value=int(roi.get("sat_max", 75)))
        val_var = tk.IntVar(value=int(roi.get("val_min", 137)))

        ctk.CTkLabel(ventana, text=f"CALIBRACIÓN DE SOLDADURA · ROI {indice + 1}",
                     font=("Arial", 19, "bold")).pack(pady=(14, 3))
        ctk.CTkLabel(ventana, text="ORIGINAL                         MÁSCARA HSV  (blanco = candidato a soldadura)",
                     font=("Arial", 12)).pack(pady=(0, 8))
        imagen_label = tk.Label(ventana, bg="#151515")
        imagen_label.pack(fill="both", expand=True, padx=16, pady=5)

        controles = ctk.CTkFrame(ventana)
        controles.pack(fill="x", padx=18, pady=(8, 5))
        controles.grid_columnconfigure(0, weight=1)
        controles.grid_columnconfigure(1, weight=1)
        etiqueta_sat = ctk.CTkLabel(controles, text="")
        etiqueta_sat.grid(row=0, column=0, pady=(8, 0))
        etiqueta_val = ctk.CTkLabel(controles, text="")
        etiqueta_val.grid(row=0, column=1, pady=(8, 0))

        def actualizar_textos():
            etiqueta_sat.configure(text=f"Saturación máxima: {sat_var.get()}")
            etiqueta_val.configure(text=f"Brillo mínimo: {val_var.get()}")

        def cambiar_sat(valor):
            sat_var.set(int(round(valor)))
            actualizar_textos()

        def cambiar_val(valor):
            val_var.set(int(round(valor)))
            actualizar_textos()

        slider_sat = ctk.CTkSlider(controles, from_=0, to=255, number_of_steps=255,
                                   command=cambiar_sat)
        slider_sat.grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 12))
        slider_val = ctk.CTkSlider(controles, from_=0, to=255, number_of_steps=255,
                                   command=cambiar_val)
        slider_val.grid(row=1, column=1, sticky="ew", padx=16, pady=(4, 12))
        slider_sat.set(sat_var.get())
        slider_val.set(val_var.get())
        actualizar_textos()

        botones = ctk.CTkFrame(ventana, fg_color="transparent")
        botones.pack(fill="x", padx=18, pady=(2, 14))
        botones.grid_columnconfigure((0, 1), weight=1)

        def cerrar(guardar=False):
            if not estado["activo"]:
                return
            estado["activo"] = False
            if estado["after"] is not None:
                try:
                    ventana.after_cancel(estado["after"])
                except tk.TclError:
                    pass
            if guardar:
                roi["sat_max"] = sat_var.get()
                roi["val_min"] = val_var.get()
                self.guardar_configuracion()
            try:
                ventana.grab_release()
            except tk.TclError:
                pass
            self.ventana_mascara = None
            ventana.destroy()

        ctk.CTkButton(botones, text="GUARDAR AJUSTES", fg_color="#218C4A",
                      command=lambda: cerrar(True)).grid(row=0, column=0, padx=6, sticky="ew")
        ctk.CTkButton(botones, text="CANCELAR", fg_color="#8B2E2E",
                      command=lambda: cerrar(False)).grid(row=0, column=1, padx=6, sticky="ew")
        ventana.protocol("WM_DELETE_WINDOW", lambda: cerrar(False))

        def refrescar():
            if not estado["activo"] or self.cerrando:
                return
            try:
                frame = self.frame_actual
                if frame is not None:
                    imagen = self.obtener_roi(frame, roi)
                    if imagen is not None and imagen.size:
                        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
                        mascara = cv2.inRange(hsv, (0, 0, val_var.get()),
                                              (179, sat_var.get(), 255))
                        # Conservar píxeles exactos de la máscara, sin interpolación.
                        ancho_disponible = max(300, imagen_label.winfo_width() - 20)
                        alto_disponible = max(150, imagen_label.winfo_height() - 20)
                        alto, ancho = imagen.shape[:2]
                        escala = min(5.0, ancho_disponible / max(2 * ancho, 1),
                                     alto_disponible / max(alto, 1))
                        tam = (max(1, int(ancho * escala)), max(1, int(alto * escala)))
                        original = cv2.resize(imagen, tam, interpolation=cv2.INTER_NEAREST)
                        binaria = cv2.resize(mascara, tam, interpolation=cv2.INTER_NEAREST)
                        combinada = np.hstack((original, cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)))
                        rgb = cv2.cvtColor(combinada, cv2.COLOR_BGR2RGB)
                        foto = ImageTk.PhotoImage(Image.fromarray(rgb))
                        estado["foto"] = foto
                        imagen_label.configure(image=foto)
                estado["after"] = ventana.after(120, refrescar)
            except (tk.TclError, RuntimeError):
                return
            except Exception as error:
                print(f"[MÁSCARA] Error al actualizar: {error}")
                if estado["activo"]:
                    estado["after"] = ventana.after(250, refrescar)

        refrescar()

    @staticmethod
    def detectar_puente_soldadura(imagen, roi):
        """Detector experimental: busca continuidad metálica entre cuatro pads.

        Los centros se calibran manualmente. Si la imagen no permite una
        segmentación inequívoca, devuelve ERROR, nunca PASS.
        """
        puntos = roi.get("pads", [])
        if len(puntos) != 4 or imagen is None or imagen.size == 0:
            return {"estado": "ERROR", "motivo": "CALIBRAR 4 PADS", "imagen_resultado": imagen}
        alto, ancho = imagen.shape[:2]
        if any(not (3 <= x < ancho-3 and 3 <= y < alto-3) for x,y in puntos):
            return {"estado": "ERROR", "motivo": "PADS FUERA DE IMAGEN", "imagen_resultado": imagen}
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        # Soldadura: reflectancia alta y saturación baja; excluir máscara verde.
        sat_max = int(np.clip(roi.get("sat_max", 75), 0, 255))
        val_min = int(np.clip(roi.get("val_min", 137), 0, 255))
        mascara = cv2.inRange(hsv, (0, 0, val_min), (179, sat_max, 255))
        # Evitar unir pads artificialmente mediante dilatación/cierre morfológico.
        cantidad, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, 8)
        ids = []
        for x,y in puntos:
            x,y = int(x), int(y)
            # Permitir un error de clic de 3 px, sin conectar regiones separadas.
            region = etiquetas[max(0,y-3):min(alto,y+4), max(0,x-3):min(ancho,x+4)]
            valores, frecuencias = np.unique(region[region > 0], return_counts=True)
            if len(valores) == 0:
                return {"estado": "ERROR", "motivo": "PAD NO DETECTADO: AJUSTAR LUZ/UMBRAL", "imagen_resultado": imagen}
            elegido = int(valores[np.argmax(frecuencias)])
            if stats[elegido, cv2.CC_STAT_AREA] < 3:
                return {"estado": "ERROR", "motivo": "PAD CON SEGMENTACIÓN DÉBIL", "imagen_resultado": imagen}
            ids.append(elegido)
        resultado = imagen.copy()
        for n,(x,y) in enumerate(puntos,1):
            cv2.circle(resultado, (int(x),int(y)), 5, (255,0,255), 1)
            cv2.putText(resultado,str(n),(int(x)+5,int(y)-4),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,0,255),1)
        conexiones = []
        for a in range(4):
            for b in range(a+1,4):
                if ids[a] == ids[b]:
                    conexiones.append((a,b))
                    cv2.line(resultado, tuple(puntos[a]), tuple(puntos[b]), (0,0,255), 3)
        if conexiones:
            return {"estado": "FAIL", "motivo": "PUENTE VISIBLE: " + ", ".join(f"{a+1}-{b+1}" for a,b in conexiones),
                    "imagen_resultado": resultado, "defectos": len(conexiones)}
        return {"estado": "PASS", "motivo": "4 PADS VISUALMENTE SEPARADOS",
                "imagen_resultado": resultado, "defectos": 0}

    def comparar_roi_con_master(self, imagen_actual, roi_config):
        masters = roi_config.get("masters", [])
        if not masters:
            return None
        candidatos = []
        for numero, master in enumerate(masters, 1):
            resultado = self._comparar_un_master(imagen_actual, roi_config, master)
            if resultado is not None:
                resultado["master_utilizado"] = numero
                candidatos.append(resultado)
        if not candidatos:
            return None
        # Primero los que cumplen todos los criterios, luego la similitud.
        return max(candidatos, key=lambda r: (r["aprobado"], r["similitud"]))

    def _comparar_un_master(self, imagen_actual, roi_config, master):
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

        # Un cierre ligero une píxeles cercanos sin eliminar primero defectos pequeños.
        # Se elimina MORPH_OPEN porque podía borrar diferencias reales detectadas
        # con sensibilidades altas.
        kernel = np.ones((3, 3), np.uint8)
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

        criterios_fail = []
        if porcentaje_similitud < roi_config["similitud_minima"]:
            criterios_fail.append("similitud")
        if area_defecto_mayor >= roi_config["area_defecto_maxima"]:
            criterios_fail.append("área máxima")
        if porcentaje_diferente >= roi_config["porcentaje_diferente_maximo"]:
            criterios_fail.append("% diferente")

        aprobado = not criterios_fail
        motivo = "OK" if aprobado else ", ".join(criterios_fail)

        return {
            "aprobado": aprobado,
            "similitud": porcentaje_similitud,
            "diferencia": porcentaje_diferente,
            "defectos": cantidad_defectos,
            "area_defecto_mayor": area_defecto_mayor,
            "umbral_diferencia": umbral_diferencia,
            "motivo": motivo,
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

            if roi.get("tipo") == "SOLDER_BRIDGE":
                imagen_roi = self.obtener_roi(frame, roi)
                deteccion = self.detectar_puente_soldadura(imagen_roi, roi)
                roi["resultado"] = deteccion["estado"]
                roi["motivo"] = deteccion["motivo"]
                roi["similitud"] = None
                roi["diferencia"] = 0.0
                roi["defectos"] = deteccion.get("defectos", 0)
                resultados_validos.append(roi["resultado"] == "PASS")
                color = (0,255,0) if roi["resultado"] == "PASS" else (0,0,255)
                anotada = deteccion.get("imagen_resultado")
                if anotada is not None and anotada.size:
                    frame_resultado[y1:y2, x1:x2] = cv2.resize(anotada, (x2-x1,y2-y1))
            elif roi.get("tipo", "INSPECCION") == "DATAMATRIX":
                imagen_roi = self.obtener_roi(frame, roi)
                codigo = self.leer_data_matrix(imagen_roi)
                roi["codigo"] = codigo or ""
                roi["similitud"] = None
                roi["diferencia"] = 0.0
                roi["defectos"] = 0
                longitud = int(roi.get("dm_longitud", 16))
                parte = str(roi.get("dm_numero_parte", "")).strip()
                configuracion_valida = bool(parte) and longitud > len(parte)
                aprobado = bool(codigo) and configuracion_valida and len(codigo) == longitud and codigo.startswith(parte)
                if not configuracion_valida:
                    roi["motivo"] = "CONFIGURAR DATAMATRIX"
                elif not codigo:
                    roi["motivo"] = "NO LEÍDO"
                elif len(codigo) != longitud:
                    roi["motivo"] = f"LONGITUD {len(codigo)}/{longitud}"
                elif not codigo.startswith(parte):
                    roi["motivo"] = "NÚMERO DE PARTE INCORRECTO"
                else:
                    roi["motivo"] = "OK"
                roi["resultado"] = "PASS" if aprobado else "FAIL"
                resultados_validos.append(aprobado)
                color = (0, 255, 0) if aprobado else (0, 0, 255)
            elif roi["master"] is None:
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
                    roi["area_defecto_mayor"] = resultado["area_defecto_mayor"]
                    roi["umbral_diferencia"] = resultado["umbral_diferencia"]
                    roi["motivo"] = resultado["motivo"]
                    roi["master_utilizado"] = resultado["master_utilizado"]
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

        masters_cargadas = sum(roi.get(
            "tipo") in ("DATAMATRIX", "SOLDER_BRIDGE") or roi["master"] is not None for roi in self.rois)

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
            alineado, motivo = self.obtener_frame_alineado(frame)
            self.label_alineacion.configure(text=motivo)
            if alineado is None:
                frame_mostrar = frame.copy()
                self.label_resultado_general.configure(text="ERROR POSICIONAMIENTO", fg_color="#B3261E")
            else:
                frame_mostrar, _ = self.procesar_frame_inspeccion(alineado)
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

        if self.area_pcb is not None and not self.mostrando_fotografia:
            cv2.rectangle(frame_mostrar, self.area_pcb[:2], self.area_pcb[2:], (255, 200, 0), 2)
        if self.roi_temporal is not None and not self.mostrando_fotografia:
            tx1, ty1, tx2, ty2 = self.roi_temporal
            cv2.rectangle(
                frame_mostrar,
                (tx1, ty1),
                (tx2, ty2),
                (255, 255, 0),
                2,
            )

        if not self.mostrando_fotografia:
            for i, roi in enumerate(self.rois):
                puntos = self.pads_temporales if self.calibrando_pads and i == self.roi_seleccionado else roi.get("pads", [])
                for n, (px, py) in enumerate(puntos, 1):
                    cx, cy = roi["x1"]+px, roi["y1"]+py
                    cv2.circle(frame_mostrar, (cx,cy), 5, (255, 0, 255), 2)
                    cv2.putText(frame_mostrar, str(n), (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,0,255), 1)
        self.mostrar_frame(frame_mostrar)
        self.actualizacion_programada = self.ventana.after(
            30, self.actualizar_video
        )

    def actualizar_etiquetas_resultados(self):
        for indice, (roi, etiqueta) in enumerate(zip(self.rois, self.labels_resultados), start=1):
            if roi.get("tipo") == "DATAMATRIX":
                codigo = roi.get("codigo", "")
                texto = (f'ROI {indice}: {roi["resultado"]} | DM: {codigo or "NO LEÍDO"} '
                         f'| {roi.get("motivo", "")}' )
            elif roi["similitud"] is None:
                texto = f'ROI {indice}: {roi["resultado"]} | {roi.get("motivo", "")}'
            else:
                area_mayor = roi.get("area_defecto_mayor", 0.0)
                umbral = roi.get("umbral_diferencia", "-")
                motivo = roi.get("motivo", "OK")
                texto = (
                    f'ROI {indice}: {roi["resultado"]} | S:{roi["similitud"]:.2f}% | '
                    f'D:{roi["diferencia"]:.3f}% | A:{area_mayor:.0f}px² | '
                    f'U:{umbral} | M:{roi.get("master_utilizado", "-")} | {motivo}'
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

        if not any(roi.get("tipo") in ("DATAMATRIX", "SOLDER_BRIDGE") or roi["master"] is not None for roi in self.rois):
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
                    "LISTO DATAMATRIX" if roi.get("tipo") == "DATAMATRIX" else
                    ("LISTO SOLDER BRIDGE" if roi.get("tipo") == "SOLDER_BRIDGE" else
                    ("LISTO" if roi["master"] is not None else "SIN MASTER"))
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

        if not any(roi.get("tipo") in ("DATAMATRIX", "SOLDER_BRIDGE") or roi["master"] is not None for roi in self.rois):
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
        alineado, motivo = self.obtener_frame_alineado(frame_capturado)
        self.label_alineacion.configure(text=motivo)
        if alineado is None:
            self.label_resultado_general.configure(text="ERROR POSICIONAMIENTO", fg_color="#B3261E")
            return
        frame_comparado, resultado_general = self.procesar_frame_inspeccion(alineado)

        fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nombre_resultado = resultado_general.replace(" ", "_")
        rois_dm = [roi for roi in self.rois if roi.get("tipo") == "DATAMATRIX"]
        if rois_dm:
            # Solo incluir un ID leído y validado. Evitar caracteres ilegales en Windows.
            roi_dm = next((r for r in rois_dm if r.get("resultado") == "PASS"), None)
            codigo = str(roi_dm.get("codigo", "")) if roi_dm else ""
            codigo_archivo = re.sub(r'[^A-Za-z0-9_-]', '_', codigo)[:150] if codigo else "DM_NO_VALIDO"
            nombre_archivo = f"{codigo_archivo}_{nombre_resultado}_{fecha}.jpg"
        else:
            nombre_archivo = f"{nombre_resultado}_{fecha}.jpg"
        ruta = os.path.join(self.carpeta_resultados, nombre_archivo)
        # Evitar sobreescribir capturas del mismo segundo.
        if os.path.exists(ruta):
            base, extension = os.path.splitext(ruta)
            consecutivo = 2
            while os.path.exists(f"{base}_{consecutivo}{extension}"):
                consecutivo += 1
            ruta = f"{base}_{consecutivo}{extension}"

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
        """Cierra cámara, callbacks y Tk en un orden seguro."""
        if self.cerrando:
            return

        print("[CIERRE] Cierre solicitado por la aplicación/usuario.")
        self.cerrando = True
        if self.ventana_mascara is not None:
            try:
                self.ventana_mascara.grab_release()
                self.ventana_mascara.destroy()
            except tk.TclError:
                pass
            self.ventana_mascara = None
        self.comparacion_activa = False
        self.mostrando_fotografia = False

        for after_id_name in ("actualizacion_programada", "after_zoom_id"):
            after_id = getattr(self, after_id_name, None)
            if after_id is not None:
                try:
                    self.ventana.after_cancel(after_id)
                except (tk.TclError, Exception):
                    pass
                setattr(self, after_id_name, None)

        if self.camara is not None:
            try:
                self.camara.release()
            except Exception:
                pass
            self.camara = None

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            if self.ventana.winfo_exists():
                self.ventana.quit()
                self.ventana.destroy()
        except tk.TclError:
            pass

    def ejecutar(self):
        """Inicia mainloop únicamente cuando la inicialización terminó bien."""
        if not self.inicializada or self.cerrando:
            return

        try:
            self.ventana.mainloop()
        except tk.TclError as error:
            # Evita el traceback secundario si Tk ya fue destruido durante el cierre.
            if "application has been destroyed" not in str(error):
                raise


if __name__ == "__main__":
    aplicacion = CamaraVision()
    if aplicacion.inicializada:
        aplicacion.ejecutar()
