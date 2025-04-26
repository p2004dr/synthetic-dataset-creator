# Synthetic Dataset Creator - Object Detection

Una herramienta para la generación de datasets sintéticos para entrenamiento de modelos de detección de objetos.

## 📋 Descripción

SyntheticObjectDetector es una herramienta flexible diseñada para generar datasets sintéticos de alta calidad para el entrenamiento, validación y prueba de modelos de detección de objetos. Al combinar imágenes de objetos con fondos variados, este proyecto permite crear grandes conjuntos de datos anotados sin el costoso proceso de etiquetado manual.

## 🌟 Características

- Generación automática de datasets en formato YOLO
- Distribución personalizable entre conjuntos de entrenamiento, validación y prueba
- Posibilidad de generar miles de imágenes únicas
- Soporte para múltiples clases de objetos
- Algoritmos para evitar solapamientos excesivos entre objetos
- Transformaciones aleatorias (escala, rotación) para aumentar la variabilidad
- Visualización de muestras para verificación

## 🛠️ Estructura del Proyecto

```
project/
├── config.py             # Configuración global
├── main.py               # Punto de entrada principal
├── utils/
│   ├── __init__.py
│   ├── file_utils.py     # Operaciones de archivos/directorios
│   ├── visualization.py  # Funciones de visualización
│   └── annotations.py    # Utilidades para formato YOLO
├── image_processing/
│   ├── __init__.py
│   ├── loaders.py        # Carga de imágenes
│   ├── transformations.py # Transformaciones de imágenes
│   └── composition.py    # Composición de imágenes
└── dataset/
    ├── __init__.py
    └── generator.py      # Generación del dataset
```

## 🚀 Instalación

1. Clona este repositorio:

```bash
git clone https://github.com/yourusername/SyntheticObjectDetector.git
cd SyntheticObjectDetector
```

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## 📋 Requisitos

- Python 3.7+
- Dependencias listadas en `requirements.txt`:
  - numpy
  - opencv-python
  - matplotlib
  - pillow

## 📊 Uso

### Preparación

1. Organiza tus imágenes de objetos en subcarpetas por clase dentro de la carpeta `card_images/`:

```
card_images/
├── class1/
│   ├── obj1.png
│   ├── obj2.png
│   └── ...
├── class2/
│   ├── obj1.png
│   └── ...
└── ...
```

2. Coloca imágenes de fondo en la carpeta `backgrounds/`:

```
backgrounds/
├── bg1.jpg
├── bg2.jpg
└── ...
```

### Configuración

Edita el archivo `config.py` para personalizar:

- Clases de objetos
- Número de imágenes a generar
- Ratios de división del dataset (entrenamiento/validación/prueba)
- Parámetros de composición (solapamiento, etc.)

### Generación del Dataset

Ejecuta el script principal:

```bash
python main.py
```

El dataset generado se almacenará en la carpeta `dataset/` con la siguiente estructura:

```
dataset/
├── train/
│   ├── images/   # Imágenes sintéticas generadas para entrenamiento
│   └── labels/   # Etiquetas YOLO correspondientes
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml     # Archivo de configuración para entrenamiento YOLO
```

## 🔄 Flujo de Trabajo

1. El generador carga las imágenes de objetos y fondos
2. Para cada imagen a generar:
   - Se selecciona un fondo aleatorio
   - Se seleccionan objetos aleatorios
   - Se aplican transformaciones (escala, rotación)
   - Se colocan objetos en posiciones aleatorias evitando solapamiento excesivo
   - Se generan las anotaciones en formato YOLO
   - Se guarda la imagen compuesta y sus etiquetas

## 📚 Recomendaciones para mejores resultados

- **Imágenes de objetos**: Utiliza PNG con fondo transparente para cada objeto individual
- **Variaciones**: Para mejores resultados, incluye múltiples variantes de cada objeto con diferentes ángulos y condiciones de iluminación
- **Fondos**: Utiliza una variedad de fondos que representen los entornos donde se encontrarán los objetos en la vida real
- **Balanceo de clases**: Asegúrate de tener un número similar de ejemplos para cada clase

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios propuestos antes de enviar un pull request.

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para más detalles.