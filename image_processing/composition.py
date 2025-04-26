"""
Funciones para componer imágenes, colocando cartas sobre fondos.
"""
import random
import cv2
import numpy as np
from utils.annotations import check_overlap
from image_processing.transformations import calculate_bounding_box

def overlay_card(background, card, position):
    """
    Sobrepone una carta en la posición especificada del fondo.
    
    Args:
        background: Imagen de fondo
        card: Diccionario con información de la carta
        position: Tupla (x_offset, y_offset) donde colocar la carta
        
    Returns:
        Tupla (xmin, ymin, xmax, ymax) representando la bounding box
    """
    x_offset, y_offset = position
    card_img = card['image']
    
    # Si la imagen de la carta está vacía o inválida, retornar una bounding box vacía
    if card_img is None or card_img.size == 0:
        return 0, 0, 0, 0
    
    # Asegurarse de que la posición esté dentro de los límites del fondo
    background_h, background_w = background.shape[:2]
    card_h, card_w = card_img.shape[:2]
    
    # Si parte de la carta se sale del fondo, ajustar la posición
    if x_offset + card_w > background_w:
        x_offset = background_w - card_w
    if y_offset + card_h > background_h:
        y_offset = background_h - card_h
    
    # Asegurarse de que los offsets no sean negativos
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)
    
    # Región de interés en el fondo
    roi = background[y_offset:y_offset+card_h, x_offset:x_offset+card_w]
    
    # Si el ROI es más pequeño que la carta, recortar la carta
    if roi.shape[0] < card_h or roi.shape[1] < card_w:
        card_img = card_img[:roi.shape[0], :roi.shape[1]]
        card_h, card_w = card_img.shape[:2]
    
    # Extraer canales alfa y color de la carta
    if card_img.shape[2] == 4:  # Si tiene canal alfa
        alpha = card_img[:, :, 3] / 255.0
        card_rgb = card_img[:, :, :3]
        
        # Crear una copia de ROI para modificar
        roi_copy = roi.copy()
        
        # Calcular imagen mezclada usando el canal alfa como máscara
        for c in range(3):  # para cada canal de color
            roi_copy[:, :, c] = roi[:, :, c] * (1 - alpha) + card_rgb[:, :, c] * alpha
        
        # Colocar el resultado de vuelta en el fondo
        background[y_offset:y_offset+card_h, x_offset:y_offset+card_h] = roi_copy
    else:
        # Si no hay transparencia, simplemente sobrescribir
        background[y_offset:y_offset+card_h, x_offset:x_offset+card_w] = card_img[:, :, :3]
    
    # Calcular bounding box
    xmin, ymin, xmax, ymax = calculate_bounding_box(card, x_offset, y_offset)
    
    return xmin, ymin, xmax, ymax

def place_cards_on_background(background, cards, label_base_percent, placed_boxes=None):
    """
    Coloca múltiples cartas sobre un fondo con una distribución aleatoria sin solapamiento excesivo.
    
    Args:
        background: Imagen de fondo
        cards: Lista de cartas disponibles para colocar
        label_base_percent: Diccionario con porcentajes base por tipo de carta
        placed_boxes: Lista opcional de cajas ya colocadas
        
    Returns:
        Lista de objetos colocados con sus etiquetas y bounding boxes
    """
    bg_height, bg_width = background.shape[:2]
    bg_area = bg_width * bg_height
    
    if placed_boxes is None:
        placed_boxes = []
    
    # Determinar una carta aleatoria para definir el número de cartas a colocar
    random_label = random.choice(list(label_base_percent.keys()))
    random_base_percent = label_base_percent[random_label]
    
    # Decidir el número de cartas en función del tamaño base
    if random_base_percent > 0.15:  # Si el tamaño base es más del 15%
        num_cards = random.randint(1, 5)
    elif random_base_percent > 0.10:  # Si el tamaño base es 15% o menos
        num_cards = random.randint(1, 7)
    elif random_base_percent > 0.05:  # Si el tamaño base es 15% o menos
        num_cards = random.randint(1, 10)
    else:  # Si el tamaño base es 15% o menos
        num_cards = random.randint(1, 12)
    
    # Para cada carta, aplicar transformaciones y colocarla
    objects = []
    attempts_per_card = 15  # Número máximo de intentos para colocar cada carta sin solapamiento excesivo
    
    for _ in range(num_cards):
        # Seleccionar una carta aleatoria
        card = random.choice(cards)
        label = card['label']
        
        # Obtener el porcentaje base para este tipo de carta
        base_percent = label_base_percent[label]
        
        # Aplicar variación aleatoria de ±20% al porcentaje base
        variation = random.uniform(0.9, 1.1)
        desired_area = base_percent * variation * bg_area
        
        # Calcular el factor de escala basado en el área deseada
        original_area = card['width'] * card['height']
        scale_factor = (desired_area / original_area) ** 0.5 if original_area > 0 else 0.1
        
        # Limitar el tamaño máximo para evitar que la carta sea más grande que el fondo
        max_scale_w = bg_width / card['width']
        max_scale_h = bg_height / card['height']
        scale_factor = min(scale_factor, max_scale_w * 0.9, max_scale_h * 0.9)
        
        # Aplicar transformaciones con el factor de escala calculado
        transformed_card = apply_transformations(card, scale_factor)
        
        # Intentar encontrar una posición sin solapamiento excesivo
        card_placed = False
        for attempt in range(attempts_per_card):
            # Elegir una posición aleatoria
            x_offset = random.randint(0, max(0, bg_width - transformed_card['width']))
            y_offset = random.randint(0, max(0, bg_height - transformed_card['height']))
            
            # Calcular bounding box preliminar
            xmin = x_offset
            ymin = y_offset
            xmax = x_offset + transformed_card['width']
            ymax = y_offset + transformed_card['height']
            
            # Verificar solapamiento con las cartas ya colocadas
            if not check_overlap((xmin, ymin, xmax, ymax), placed_boxes):
                # Superponer la carta y obtener la bounding box real
                xmin, ymin, xmax, ymax = overlay_card(background, transformed_card, (x_offset, y_offset))
                
                # Verificar que la bounding box es válida (área > 0)
                if xmin < xmax and ymin < ymax:
                    # Añadir a la lista de objetos
                    placed_boxes.append((xmin, ymin, xmax, ymax))
                    
                    objects.append({
                        'label': transformed_card['label'],
                        'xmin': max(0, xmin),
                        'ymin': max(0, ymin),
                        'xmax': min(bg_width, xmax),
                        'ymax': min(bg_height, ymax)
                    })
                    
                    card_placed = True
                    break
        
        # Si no se pudo colocar la carta después de todos los intentos, continuar con la siguiente
        if not card_placed and attempt == attempts_per_card - 1:
            print(f"No se pudo colocar una carta '{label}' sin solapamiento excesivo.")
    
    return objects

def generate_label_base_percents(cards):
    """
    Genera porcentajes base aleatorios para cada tipo de carta.
    
    Args:
        cards: Lista de cartas para extraer las etiquetas únicas
        
    Returns:
        Diccionario con porcentajes base por tipo de carta
    """
    label_base_percent = {}
    
    for card in cards:
        label = card['label']
        if label not in label_base_percent:
            label_base_percent[label] = random.uniform(0.025, 0.25)
    
    return label_base_percent