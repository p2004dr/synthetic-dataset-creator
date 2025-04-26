"""
Funciones para generar el dataset completo.
"""
import os
import cv2
import random
from image_processing.loaders import load_card_images, load_backgrounds
from image_processing.composition import generate_label_base_percents, place_cards_on_background
from utils.file_utils import generate_unique_filename, create_data_yaml
from utils.annotations import create_yolo_annotation, save_yolo_annotation

def generate_synthetic_image(cards, backgrounds, output_images_dir, output_labels_dir, image_index, label_to_id):
    """
    Genera una imagen sintética con cartas aleatorias sobre un fondo aleatorio.
    
    Args:
        cards: Lista de cartas disponibles
        backgrounds: Lista de fondos disponibles
        output_images_dir: Directorio de salida para imágenes
        output_labels_dir: Directorio de salida para etiquetas
        image_index: Índice de la imagen para identificación única
        label_to_id: Diccionario para mapear etiquetas a IDs de clase
        
    Returns:
        Nombre del archivo de imagen generado
    """
    # Seleccionar un fondo aleatorio
    background = random.choice(backgrounds).copy()
    bg_height, bg_width = background.shape[:2]
    
    # Generar porcentajes base para cada tipo de carta
    label_base_percent = generate_label_base_percents(cards)
    
    # Colocar cartas sobre el fondo
    objects = place_cards_on_background(background, cards, label_base_percent)
    
    # Crear un nombre de archivo único
    image_filename = generate_unique_filename(prefix=f"IMG_{image_index}_", extension=".jpg")
    
    # Guardar la imagen
    image_path = os.path.join(output_images_dir, image_filename)
    cv2.imwrite(image_path, background)
    
    # Crear y guardar anotación YOLO
    annotation_lines = create_yolo_annotation(objects, bg_width, bg_height, label_to_id)
    label_filename = f"{os.path.splitext(image_filename)[0]}.txt"
    label_path = os.path.join(output_labels_dir, label_filename)
    
    save_yolo_annotation(annotation_lines, label_path)
    
    return image_filename

def generate_dataset(config):
    """
    Genera el dataset completo según la configuración proporcionada.
    
    Args:
        config: Diccionario con la configuración del dataset
        
    Returns:
        Número total de imágenes generadas
    """
    # Cargar recursos necesarios
    cards = load_card_images(config['card_images_dir'], config['classes'], config['group_variations'])
    if not cards:
        print("Error: No se encontraron imágenes de cartas en el directorio.")
        return 0
    
    backgrounds = load_backgrounds(config['backgrounds_dir'], config['target_size'])
    if not backgrounds:
        print("Error: No se encontraron imágenes de fondo en el directorio.")
        return 0
    
    print(f"Se cargaron {len(cards)} cartas y {len(backgrounds)} fondos")
    
    # Crear diccionario de mapeo de etiquetas a IDs
    label_to_id = {label: i for i, label in enumerate(config['classes'])}
    
    # Calcular número de imágenes para cada conjunto
    total_images = config['total_images']
    num_train = int(total_images * config['train_ratio'])
    num_valid = int(total_images * config['valid_ratio'])
    num_test = total_images - num_train - num_valid
    
    # Generar imágenes de entrenamiento
    print(f"Generando {num_train} imágenes para entrenamiento...")
    for i in range(num_train):
        if i % 10 == 0:
            print(f"Progreso: {i}/{num_train}")
        generate_synthetic_image(
            cards, 
            backgrounds, 
            config['train_images_dir'], 
            config['train_labels_dir'], 
            i,
            label_to_id
        )
    
    # Generar imágenes de validación
    print(f"Generando {num_valid} imágenes para validación...")
    for i in range(num_valid):
        if i % 10 == 0:
            print(f"Progreso: {i}/{num_valid}")
        generate_synthetic_image(
            cards, 
            backgrounds, 
            config['valid_images_dir'], 
            config['valid_labels_dir'], 
            i + num_train,
            label_to_id
        )
    
    # Generar imágenes de test
    print(f"Generando {num_test} imágenes para test...")
    for i in range(num_test):
        if i % 10 == 0:
            print(f"Progreso: {i}/{num_test}")
        generate_synthetic_image(
            cards, 
            backgrounds, 
            config['test_images_dir'], 
            config['test_labels_dir'], 
            i + num_train + num_valid,
            label_to_id
        )
    
    # Crear archivo data.yaml
    create_data_yaml(config['output_dir'], config['classes'])
    
    return total_images