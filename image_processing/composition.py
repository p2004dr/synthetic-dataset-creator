"""
Funciones para componer imágenes, colocando cartas sobre fondos.
"""
import random
import cv2
import numpy as np
from utils.annotations import check_overlap, check_image_coverage
from image_processing import apply_transformations
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
        background[y_offset:y_offset+card_h, x_offset:x_offset+card_w] = roi_copy
    else:
        # Si no hay transparencia, simplemente sobrescribir
        background[y_offset:y_offset+card_h, x_offset:x_offset+card_w] = card_img[:, :, :3]
    
    xmin_list, ymin_list, xmax_list, ymax_list = [], [], [], []
    for bbox in card['bounding_boxes']:
        xmin_list.append(bbox['xmin'] + x_offset)
        ymin_list.append(bbox['ymin'] + y_offset)
        xmax_list.append(bbox['xmax'] + x_offset)
        ymax_list.append(bbox['ymax'] + y_offset)

    return xmin_list, ymin_list, xmax_list, ymax_list

def place_cards_on_background(background, cards, label_base_percent, placed_boxes=None):
    """
    Coloca múltiples cartas sobre un fondo con una distribución aleatoria sin solapamiento excesivo.
    
    Args:
        background: Imagen de fondo
        cards: Lista de cartas disponibles para colocar (cada una con 'width', 'height' y 'bounding_boxes')
        label_base_percent: Diccionario con porcentajes base por tipo de carta
        placed_boxes: Lista opcional de tuplas (xmin, ymin, xmax, ymax) ya colocadas
        
    Returns:
        Lista de objetos colocados con sus etiquetas y bounding boxes
    """
    bg_h, bg_w = background.shape[:2]
    bg_area = bg_h * bg_w

    if placed_boxes is None:
        placed_boxes = []

    # Elegir un label al azar sólo para decidir cuántas cartas colocar
    random_label = random.choice(list(label_base_percent.keys()))
    random_base_percent = label_base_percent[random_label]

    # Decidir cuántas cartas colocar según el porcentaje
    if random_base_percent > 0.15:
        num_cards = random.randint(1, 5)
    elif random_base_percent > 0.10:
        num_cards = random.randint(1, 7)
    elif random_base_percent > 0.05:
        num_cards = random.randint(1, 10)
    else:
        num_cards = random.randint(1, 12)

    objects = []
    attempts_per_card = 15

    for _ in range(num_cards):
        card = random.choice(cards)
        label = card['bounding_boxes'][0]['label']  # asumimos que al menos hay 1 bbox
        base_percent = label_base_percent.get(label, 0.05)

        # Calcular scale_factor para mantener el área aproximada
        variation = random.uniform(0.9, 1.1)
        desired_area = base_percent * variation * bg_area
        orig_area = card['width'] * card['height']
        scale_factor = (desired_area / orig_area)**0.5 if orig_area > 0 else 0.1

        # No dejar que la carta exceda el fondo
        max_sf_w = bg_w / card['width']
        max_sf_h = bg_h / card['height']
        scale_factor = min(scale_factor, max_sf_w * 0.9, max_sf_h * 0.9)

        # Aplicar transformaciones (redimensionar, rotar, etc.) que ajusten también las bboxes
        transformed = apply_transformations(card, scale_factor)

        for _ in range(attempts_per_card):
            x_off = random.randint(0, max(0, bg_w - transformed['width']))
            y_off = random.randint(0, max(0, bg_h - transformed['height']))

            # Generar todas las cajas nuevas desplazadas
            new_boxes = [
                (b['xmin'] + x_off, b['ymin'] + y_off, b['xmax'] + x_off, b['ymax'] + y_off)
                for b in transformed['bounding_boxes']
            ]
            # 1) Chequeo extra: la carta completa no puede tapar >40% de ninguna caja existente
            full_box = (x_off, y_off,
                        x_off + transformed['width'],
                        y_off + transformed['height'])
            
            if check_image_coverage(full_box, placed_boxes, max_coverage_ratio=0.4):
                continue

            # 2) Chequeo original: las cajas definidas no tapan >40% de las existentes
            if not check_overlap(new_boxes, placed_boxes):
                # Pegar la carta y obtener de nuevo las coordenadas (por si clipping)
                xmin_list, ymin_list, xmax_list, ymax_list = overlay_card(
                    background, transformed, (x_off, y_off)
                )

                # Validar cada bbox y añadirla al resultado
                for idx, b in enumerate(transformed['bounding_boxes']):
                    xb = (
                        xmin_list[idx],
                        ymin_list[idx],
                        xmax_list[idx],
                        ymax_list[idx]
                    )
                    # bbox válida
                    if xb[0] < xb[2] and xb[1] < xb[3]:
                        placed_boxes.append(xb)
                        objects.append({
                            'label': b['label'],
                            'xmin': max(0, xb[0]),
                            'ymin': max(0, xb[1]),
                            'xmax': min(bg_w, xb[2]),
                            'ymax': min(bg_h, xb[3])
                        })
                break  # carta colocada, pasamos a la siguiente

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
            if random.random() < 0.75:
                label_base_percent[label] = random.uniform(0.017, 0.20)
            elif random.random() < 0.75:
                label_base_percent[label] = random.uniform(0.020, 0.4)
            else:
                label_base_percent[label] = random.uniform(0.4, 1)
    
    return label_base_percent