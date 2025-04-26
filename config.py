"""
Configuración global para la generación de datasets de cartas de sushi.
"""
import os

# Si es True, tratar filenames que terminen en _<número> como variaciones
GROUP_VARIATIONS = True

# Tamaño fijo para todas las imágenes
TARGET_SIZE = (620, 620)

# Configuración de directorios
CARD_IMAGES_DIR = 'card_images'  # Carpeta con imágenes de cartas
BACKGROUNDS_DIR = 'backgrounds'  # Carpeta con imágenes de fondo
OUTPUT_DIR = 'dataset'  # Carpeta raíz del dataset

# Subdirectorios según la estructura solicitada
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VALID_DIR = os.path.join(OUTPUT_DIR, 'valid')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test')

# Directorios de imágenes y etiquetas
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, 'labels')
VALID_IMAGES_DIR = os.path.join(VALID_DIR, 'images')
VALID_LABELS_DIR = os.path.join(VALID_DIR, 'labels')
TEST_IMAGES_DIR = os.path.join(TEST_DIR, 'images')
TEST_LABELS_DIR = os.path.join(TEST_DIR, 'labels')

# Lista de clases en el orden solicitado
CLASSES = [
    'egg_nigiri', 'salmon_nigiri', 'squid_nigiri', 'wasabi', 
    'tempura', 'sashimi', 'dumpling', 'chopsticks', 'pudding', 'maki_roll'
]

# Crear un diccionario para mapear los nombres de etiquetas a IDs de clase
LABEL_TO_ID = {label: i for i, label in enumerate(CLASSES)}

# Parámetros de generación de datasets
TOTAL_IMAGES = 100
TRAIN_RATIO = 0.75
VALID_RATIO = 0.15
TEST_RATIO = 0.15  # Será ajustado al valor restante en el código

# Parámetros de solapamiento
MAX_OVERLAP_RATIO = 0.5
MAX_COVERAGE_RATIO = 0.8
ATTEMPTS_PER_CARD = 15

# Crear las variables que se importan en main.py
DATASET_CONFIG = {
    'total_images': TOTAL_IMAGES,
    'train_ratio': TRAIN_RATIO,
    'valid_ratio': VALID_RATIO,
    'test_ratio': TEST_RATIO,
    'group_variations': GROUP_VARIATIONS,
    'target_size': TARGET_SIZE,
    'max_overlap_ratio': MAX_OVERLAP_RATIO,
    'max_coverage_ratio': MAX_COVERAGE_RATIO,
    'attempts_per_card': ATTEMPTS_PER_CARD,
    'classes': CLASSES,
    'label_to_id': LABEL_TO_ID,
    'train_images_dir': TRAIN_IMAGES_DIR,
    'train_labels_dir': TRAIN_LABELS_DIR,
    'valid_images_dir': VALID_IMAGES_DIR,
    'valid_labels_dir': VALID_LABELS_DIR,
    'test_images_dir': TEST_IMAGES_DIR,
    'test_labels_dir': TEST_LABELS_DIR,
    'card_images_dir': CARD_IMAGES_DIR,
    'backgrounds_dir': BACKGROUNDS_DIR
}

# Definir los directorios que deben crearse
DIRECTORIES = [
    OUTPUT_DIR,
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    TRAIN_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VALID_IMAGES_DIR,
    VALID_LABELS_DIR,
    TEST_IMAGES_DIR,
    TEST_LABELS_DIR
]