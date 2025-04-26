"""
Funciones para cargar imágenes de cartas y fondos.
"""
import os
import cv2
import re
import numpy as np

def load_card_images(card_images_dir, classes, group_variations=True):
    """
    Carga imágenes de cartas y, opcionalmente, agrupa variaciones bajo un mismo label.
    
    Args:
        card_images_dir: Directorio con imágenes de cartas
        classes: Lista de clases válidas
        group_variations: Si es True, agrupar variaciones (_X) bajo la misma etiqueta
        
    Returns:
        Lista de diccionarios con información de cada carta
    """
    cards = []
    for filename in os.listdir(card_images_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Extraer nombre base sin extensión
        name_no_ext = os.path.splitext(filename)[0]

        # Agrupar variaciones si corresponde
        if group_variations:
            m = re.match(r'(.+?)(?:_[0-9]+)$', name_no_ext)
            label = m.group(1) if m else name_no_ext
        else:
            label = name_no_ext

        # Verificar si la etiqueta está en la lista de clases
        if label not in classes:
            print(f"[WARN] Etiqueta '{label}' no está en la lista de clases. Ignorando: {filename}")
            continue

        image_path = os.path.join(card_images_dir, filename)
        ext = filename.lower().split('.')[-1]
        card_img = None

        if ext == 'png':
            # Leer PNG con posible canal alfa
            tmp = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if tmp is None:
                print(f"[WARN] No se pudo leer la carta PNG: {image_path}")
                continue

            # Asegurar 4 canales BGRA
            if tmp.ndim == 2:
                continue  # imagen inválida

            if tmp.shape[2] == 4:
                card_img = tmp
            elif tmp.shape[2] == 3:
                b, g, r = cv2.split(tmp)
                alpha = np.ones(b.shape, dtype=b.dtype) * 255
                card_img = cv2.merge((b, g, r, alpha))
        else:
            # Leer JPG/JPEG y crear canal alfa opaco
            tmp = cv2.imread(image_path)
            if tmp is None:
                print(f"[WARN] No se pudo leer la carta JPEG: {image_path}")
                continue
            b, g, r = cv2.split(tmp)
            alpha = np.ones(b.shape, dtype=b.dtype) * 255
            card_img = cv2.merge((b, g, r, alpha))

        if card_img is None:
            print(f"[WARN] Imagen inválida o corrupta: {image_path}")
            continue

        cards.append({
            'image': card_img,
            'label': label,
            'width': card_img.shape[1],
            'height': card_img.shape[0]
        })

    return cards

def load_backgrounds(backgrounds_dir, target_size):
    """
    Carga todas las imágenes de fondo y las redimensiona al tamaño objetivo.
    
    Args:
        backgrounds_dir: Directorio con imágenes de fondo
        target_size: Tamaño objetivo (width, height)
        
    Returns:
        Lista de imágenes de fondo
    """
    backgrounds = []
    for filename in os.listdir(backgrounds_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(backgrounds_dir, filename)
            background = cv2.imread(img_path)
            if background is None:
                print(f"[WARN] No se pudo leer el fondo: {img_path}")
                continue
            
            # Redimensionar al tamaño objetivo
            background = cv2.resize(background, target_size, interpolation=cv2.INTER_AREA)
            backgrounds.append(background)
    return backgrounds