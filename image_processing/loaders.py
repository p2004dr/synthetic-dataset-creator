"""
Funciones para cargar imágenes de cartas y fondos.
"""
import os
import cv2
import numpy as np
import glob
from utils.file_utils import normalize_filename

def load_card_images(card_images_dir, classes, group_variations=True):
    """
    Carga imágenes de cartas con sus bounding boxes desde los archivos TXT asociados.
    
    Args:
        card_images_dir: Directorio que contiene las imágenes de las cartas
        classes: Lista de clases a incluir
        group_variations: Si se deben agrupar las variaciones de la misma carta
        
    Returns:
        Lista de diccionarios con información de cada carta
    """
    cards = []
    
    # Verificar que el directorio existe
    if not os.path.exists(card_images_dir):
        print(f"¡Error! El directorio {card_images_dir} no existe.")
        return cards
    
    # Cargar todas las imágenes PNG o JPG en el directorio
    image_paths = glob.glob(os.path.join(card_images_dir, "*.png"))
    image_paths.extend(glob.glob(os.path.join(card_images_dir, "*.jpg")))
    
    for image_path in image_paths:
        # Obtener nombre base y extensión
        filename = os.path.basename(image_path)
        base_name, extension = os.path.splitext(filename)
        
        # Verificar si existe un archivo TXT asociado a esta imagen
        txt_path = os.path.join(card_images_dir, f"{base_name}.txt")
        if not os.path.exists(txt_path):
            # Si no hay archivo TXT, omitir esta imagen
            print(f"Advertencia: No se encontró archivo TXT para {filename}, omitiendo.")
            continue
        
        # Identificar la etiqueta a partir del nombre del archivo
        # Asumimos que el formato es label_name_X.png o label_name.png
        # donde X es un número opcional para variaciones
        label = normalize_filename(base_name, group_variations)
        
        # Verificar si la etiqueta está en la lista de clases
        valid_class_found = False
        for class_name in classes:
            if label.startswith(class_name):
                valid_class_found = True
                break
        
        if not valid_class_found:
            print(f"Advertencia: La imagen {filename} no corresponde a una clase válida, omitiendo.")
            continue
        
        # Cargar la imagen con transparencia (4 canales)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error al cargar la imagen {filename}, omitiendo.")
            continue
        
        # Si la imagen no tiene canal alfa, añadirlo
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255
            img = cv2.merge((b, g, r, alpha))
        
        # Cargar las bounding boxes desde el archivo TXT
        bounding_boxes = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:  # Formato YOLO: class_id x_center y_center width height
                    try:
                        class_id = int(parts[0])
                        if class_id < len(classes):  # Verificar que el ID de clase es válido
                            class_name = classes[class_id]
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convertir de formato YOLO normalizado a píxeles
                            img_height, img_width = img.shape[:2]
                            x_center_pixels = x_center * img_width
                            y_center_pixels = y_center * img_height
                            width_pixels = width * img_width
                            height_pixels = height * img_height
                            
                            # Calcular esquinas de la bounding box
                            xmin = int(x_center_pixels - width_pixels / 2)
                            ymin = int(y_center_pixels - height_pixels / 2)
                            xmax = int(x_center_pixels + width_pixels / 2)
                            ymax = int(y_center_pixels + height_pixels / 2)
                            
                            bounding_boxes.append({
                                'label': class_name,
                                'xmin': xmin,
                                'ymin': ymin,
                                'xmax': xmax,
                                'ymax': ymax,
                                'width': xmax - xmin,
                                'height': ymax - ymin
                            })
                    except (ValueError, IndexError) as e:
                        print(f"Error al procesar línea en {txt_path}: {line.strip()} - {e}")
        
        # Si no hay bounding boxes válidas, omitir la imagen
        if not bounding_boxes:
            print(f"Advertencia: No se encontraron bounding boxes válidas en {txt_path}, omitiendo.")
            continue
        
        # Añadir a la lista de cartas
        cards.append({
            'image': img,
            'filename': filename,
            'label': label,
            'width': img.shape[1],
            'height': img.shape[0],
            'bounding_boxes': bounding_boxes
        })
    
    print(f"Se cargaron {len(cards)} cartas con bounding boxes válidas.")
    return cards

def load_backgrounds(backgrounds_dir, target_size=None):
    """
    Carga imágenes de fondo desde un directorio.
    
    Args:
        backgrounds_dir: Directorio que contiene las imágenes de fondo
        target_size: Tamaño objetivo para redimensionar las imágenes (ancho, alto)
        
    Returns:
        Lista de imágenes de fondo
    """
    backgrounds = []
    
    # Verificar que el directorio existe
    if not os.path.exists(backgrounds_dir):
        print(f"¡Error! El directorio {backgrounds_dir} no existe.")
        return backgrounds
    
    # Cargar todas las imágenes JPG, PNG y JPEG en el directorio
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(backgrounds_dir, ext)))
    
    for image_path in image_paths:
        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error al cargar la imagen de fondo {image_path}")
            continue
        
        # Redimensionar si se especifica un tamaño objetivo
        if target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        backgrounds.append(img)
    
    return backgrounds