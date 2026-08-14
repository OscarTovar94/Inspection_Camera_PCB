# Inspection Camera PCB
# By: Oscar Tovar 
# ELRAD Electronics México

# Introducción
Este script tiene la finalidad de utilizar una camara comercial, como sensor de visión para detectar defectos en PCBs como; Falta de componentes, Cortos de soldadura, Faltantes de soldadura, etc.

# Historial de Revisiones
Rev. 1.0 - 21/07/2026
- Creación del scrip main_inspection_pcb.py

# Fujo de Operación
                         ┌──────────────────────┐
                         │       INICIO         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │ Cargar config.ini            │
                    │ - Cámara                     │
                    │ - Contraseña administrador   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │ SELECCIONAR PROGRAMA            │
                  │                                 │
                  │ • Abrir programa existente      │
                  │ • Crear programa nuevo          │
                  │ • Eliminar programa             │
                  └───────────────┬─────────────────┘
                                  │
              ┌───────────────────┼─────────────────────┐
              │                   │                     │
              ▼                   ▼                     ▼
     ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐
     │ ABRIR PROGRAMA │  │ CREAR PROGRAMA  │  │ELIMINAR PROGR. │
     └───────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                    │
             │                    ▼                    ▼
             │             ┌───────────────┐    ┌───────────────┐
             │             │ Pedir clave   │    │ Pedir clave   │
             │             │ administrador │    │ administrador │
             │             └───────┬───────┘    └───────┬───────┘
             │                     │                    │
             │               ¿Clave correcta?     ¿Clave correcta?
             │                /          \           /          \
             │              NO            SÍ       NO            SÍ
             │              │              │       │              │
             │              ▼              ▼       ▼              ▼
             │        Acceso denegado   Crear   Acceso        Confirmar
             │                          programa denegado      eliminación
             │                              │                    │
             └──────────────────────────────┴────────────────────┘
                                            │
                                            ▼
                         ┌─────────────────────────────────┐
                         │ CARGAR PROGRAMA                 │
                         │ - Número de ROIs                │
                         │ - Coordenadas                   │
                         │ - Sensibilidad                  │
                         │ - Similitud mínima              │
                         │ - Masters                       │
                         └───────────────┬─────────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │ INICIAR CÁMARA            │
                           │ MODO OPERADOR ACTIVO      │
                           └────────────┬──────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 │                                             │
                 ▼                                             ▼
        ┌─────────────────────┐                     ┌──────────────────────┐
        │ OPERACIÓN NORMAL    │                     │ MODO ADMINISTRADOR   │
        └──────────┬──────────┘                     └──────────┬───────────┘
                   │                                           │
                   │                                  Presionar botón
                   │                                  MODO ADMINISTRADOR
                   │                                           │
                   │                                           ▼
                   │                                   ┌───────────────┐
                   │                                   │ Pedir clave   │
                   │                                   └───────┬───────┘
                   │                                           │
                   │                                    ¿Clave correcta?
                   │                                     /           \
                   │                                   NO             SÍ
                   │                                   │               │
                   │                                   ▼               ▼
                   │                             Mantener modo    Habilitar:
                   │                               operador       • ROI
                   │                                              • Masters
                   │                                              • Similitud
                   │                                              • Sensibilidad
                   │
                   ▼
       ┌─────────────────────────┐
       │ ¿Qué desea hacer?       │
       └───────────┬─────────────┘
                   │
          ┌────────┴─────────┐
          │                  │
          ▼                  ▼
 ┌────────────────┐   ┌──────────────────┐
 │ VER EN VIVO    │   │ INICIAR PRUEBA  │
 └───────┬────────┘   └────────┬─────────┘
         │                     │
         ▼                     ▼
 ¿Existen Masters?     ¿Cámara disponible?
    /       \              /        \
   NO       SÍ            NO         SÍ
   │         │             │          │
   ▼         ▼             ▼          ▼
 Mostrar   Comparar      Mostrar    ¿Existen Masters?
 "SIN      continuamente error       /        \
 MASTERS"  cada ROI                 NO         SÍ
             │                       │          │
             ▼                       ▼          ▼
      ┌───────────────┐         "SIN       Capturar
      │ ROI 1 PASS/FAIL│        MASTERS"    fotografía
      │ ROI 2 PASS/FAIL│                       │
      │ ROI N PASS/FAIL│                       ▼
      └───────┬────────┘                Comparar todos
              │                         los ROIs contra
              ▼                         sus Masters
       Resultado general                      │
        PASS / FAIL                           ▼
                                      ┌─────────────────┐
                                      │ ¿Todos los ROI  │
                                      │ cumplen límites?│
                                      └────────┬────────┘
                                               │
                                          ┌────┴────┐
                                          │         │
                                         SÍ        NO
                                          │         │
                                          ▼         ▼
                                        PASS       FAIL
                                          │         │
                                          └────┬────┘
                                               │
                                               ▼
                                     Guardar fotografía
                                     en resultados del
                                     programa seleccionado
                                               │
                                               ▼
                                    Mostrar imagen congelada
                                    con resultado y fecha
                                               │
                                               ▼
                                    Presionar VER EN VIVO
                                               │
                                               ▼
                                       Nueva inspección





# Arquitectura
                         ┌─────────────────────────────┐
                         │        OPERADOR / ADMIN     │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      INTERFAZ GRÁFICA       │
                         │     CustomTkinter / Tk      │
                         │                             │
                         │ • Selección de programa     │
                         │ • Modo operador             │
                         │ • Modo administrador        │
                         │ • Selección de ROI          │
                         │ • Ver en vivo               │
                         │ • Iniciar prueba            │
                         └──────────────┬──────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 │                      │                      │
                 ▼                      ▼                      ▼
      ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
      │ GESTIÓN DE         │  │ CONFIGURACIÓN      │  │ CONTROL DE CÁMARA  │
      │ PROGRAMAS          │  │ GENERAL            │  │                    │
      │                    │  │                    │  │ OpenCV             │
      │ • Crear programa   │  │ config.ini         │  │ VideoCapture       │
      │ • Abrir programa   │  │                    │  │                    │
      │ • Eliminar         │  │ • Cámara           │  │ • Captura frames   │
      │ • Password admin   │  │ • Resolución       │  │ • Imagen en vivo   │
      └─────────┬──────────┘  │ • Password admin   │  │ • Fotografía       │
                │             └────────────────────┘  └──────────┬─────────┘
                │                                                │
                ▼                                                ▼
      ┌────────────────────┐                          ┌─────────────────────┐
      │ PROGRAMA DE        │                          │ PROCESAMIENTO DE    │
      │ INSPECCIÓN         │                          │ IMAGEN              │
      │                    │                          │                     │
      │ programa.ini       │                          │ OpenCV + NumPy      │
      │                    │                          │ scikit-image        │
      │ • Cantidad ROIs    │                          │                     │
      │ • Coordenadas      │                          │ • Recorte ROI       │
      │ • Sensibilidad     │                          │ • Escala de grises  │
      │ • Similitud mínima │                          │ • Gaussian Blur     │
      │ • Área mínima      │                          │ • SSIM              │
      │ • Límites defecto  │                          │ • Umbral            │
      └─────────┬──────────┘                          │ • Morfología        │
                │                                     │ • Contornos         │
                │                                     └──────────┬──────────┘
                │                                                │
                ▼                                                ▼
       ┌────────────────────┐                         ┌──────────────────────┐
       │ MASTERS DEL        │                         │ MOTOR DE DECISIÓN    │
       │ PROGRAMA           │                         │                      │
       │                    │                         │ Por cada ROI:        │
       │ ROI_1.jpg          │◄────────────────────────│ • Similitud          │
       │ ROI_2.jpg          │                         │ • % diferencia       │
       │ ROI_3.jpg          │                         │ • Área de defecto    │
       │ ...                │                         │ • Nº defectos        │
       └────────────────────┘                         │                      │
                                                    │ PASS / FAIL por ROI  │
                                                    └───────────┬──────────┘
                                                                │
                                                                ▼
                                                    ┌──────────────────────┐
                                                    │ RESULTADO GENERAL    │
                                                    │                      │
                                                    │ Todos PASS → PASS    │
                                                    │ Algún FAIL → FAIL    │
                                                    │ Falta Master → Aviso │
                                                    └───────────┬──────────┘
                                                                │
                                                                ▼
                                                    ┌──────────────────────┐
                                                    │ ALMACENAMIENTO       │
                                                    │                      │
                                                    │ resultados/          │
                                                    │                      │
                                                    │ • Fotografía         │
                                                    │ • PASS / FAIL        │
                                                    │ • Fecha y hora       │
                                                    │ • ROIs marcados      │
                                                    └──────────────────────┘

# Arquitectura de carpetas
Inspection_Camera_PCB/
│
├── main_inspection_pcb.py
│
├── config.ini
│
└── programas/
    │
    ├── program_1/
    │   ├── programa.ini
    │   │
    │   ├── masters/
    │   │   ├── roi_1.jpg
    │   │   ├── roi_2.jpg
    │   │   └── roi_3.jpg
    │   │
    │   └── resultados/
    │       ├── PASS_yyyy-mm-dd_hh_mm_ss.jpg
    │       └── FAIL_yyyy-mm-dd_hh_mm_ss.jpg
    │
    ├── program_2/
    │   ├── programa.ini
    │   ├── masters/
    │   └── resultados/
    │
    └── program_3/
        ├── programa.ini
        ├── masters/
        └── resultados/
    .
    .
    .
    └── program_x/
        ├── programa.ini
        ├── masters/
        └── resultados/

# Mapa conceptual
config.ini
     │
     └── Configuración global
             │
             ▼
       Aplicación Python
             │
             ▼
     Selección de programa
             │
             ▼
       programa.ini
             │
             ├── Cantidad ROIs
             ├── Coordenadas
             ├── Sensibilidad
             └── Límites
             │
             ▼
          Cámara
             │
             ▼
          Frame
             │
       ┌─────┴─────┐
       ▼           ▼
    ROI actual    Masters
       │           │
       └─────┬─────┘
             ▼
      Comparación SSIM
             │
             ▼
     Análisis diferencias
             │
             ▼
        PASS / FAIL
             │
             ▼
      Guardar evidencia

#