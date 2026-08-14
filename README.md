# Inspection Camera PCB
# By: Oscar Tovar 
# ELRAD Electronics México

# Introducción
Este script tiene la finalidad de utilizar una camara comercial, como sensor de visión para detectar defectos en PCBs como; Falta de componentes, Cortos de soldadura, Faltantes de soldadura, etc.

# Historial de Revisiones
Rev. 1.0 - 21/07/2026
- Creación del scrip main_inspection_pcb.py

# Diagrama de flujo
Ejecutar: main_inspection_pcb.py
    ↓
    Ventana: Seleccionar programa
        ↓

# Arquitectura
CAMARA
   ↓
SELECCIONAR PROGRAMA
   ↓
CARGAR programa.ini
   ↓
Cantidad ROI
   ↓
Crear ROIs dinámicamente
   ↓
Cargar coordenadas
   ↓
Cargar sensibilidad de cada ROI
   ↓
Cargar masters del programa
   ↓
INSPECCIÓN
   ↓
Guardar resultado en carpeta del programa
