"""
Funciones para aplicar transformaciones a imágenes.
"""
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps

def apply_perspective_transform(image, intensity=0.2):
    """
    Aplica una transformación de perspectiva aleatoria a la imagen.
    
    Args:
        image: Imagen con 4 canales (BGRA)
        intensity: Intensidad de la deformación (0-1)
        
    Returns:
        Imagen transformada y matriz de transformación
    """
    # Separar canales
    b, g, r, a = cv2.split(image)
    rgb_img = cv2.merge([r, g, b])
    alpha = a.copy()
    
    # Obtener dimensiones
    h, w = rgb_img.shape[:2]
    
    # Puntos originales (las cuatro esquinas de la imagen)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Calcular desplazamiento máximo basado en la intensidad
    max_offset = min(w, h) * intensity
    
    # Generar puntos de destino con desplazamientos aleatorios
    # Cada esquina se mueve una cantidad aleatoria dentro del rango de intensidad
    top_left_offset = (random.uniform(-max_offset/2, max_offset/2), 
                       random.uniform(-max_offset/2, max_offset/2))
    top_right_offset = (random.uniform(-max_offset/2, max_offset/2), 
                       random.uniform(-max_offset/2, max_offset/2))
    bottom_left_offset = (random.uniform(-max_offset/2, max_offset/2), 
                         random.uniform(-max_offset/2, max_offset/2))
    bottom_right_offset = (random.uniform(-max_offset/2, max_offset/2), 
                          random.uniform(-max_offset/2, max_offset/2))
    
    # Generar puntos de destino
    pts2 = np.float32([
        [0 + top_left_offset[0], 0 + top_left_offset[1]],
        [w + top_right_offset[0], 0 + top_right_offset[1]],
        [0 + bottom_left_offset[0], h + bottom_left_offset[1]],
        [w + bottom_right_offset[0], h + bottom_right_offset[1]]
    ])
    
    # Calcular matriz de transformación de perspectiva
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # Aplicar transformación a la imagen RGB
    warped_rgb = cv2.warpPerspective(rgb_img, M, (w, h), 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(0, 0, 0))
    
    # Aplicar transformación al canal alfa
    warped_alpha = cv2.warpPerspective(alpha, M, (w, h), 
                                      borderMode=cv2.BORDER_CONSTANT, 
                                      borderValue=0)
    
    # Combinar canales transformados
    r, g, b = cv2.split(warped_rgb)
    warped_img = cv2.merge([b, g, r, warped_alpha])
    
    return warped_img, M

def apply_transformations(card, scale_factor):
    """
    Aplica transformaciones aleatorias con un factor de escala específico.
    
    Args:
        card: Diccionario con información de la carta
        scale_factor: Factor de escala a aplicar
        
    Returns:
        Carta transformada como diccionario
    """
    img = card['image'].copy()
    
    # Convertir a PIL para algunas transformaciones
    # Asumimos que img tiene 4 canales (BGRA)
    b, g, r, a = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])
    alpha = Image.fromarray(a)
    pil_img = Image.fromarray(rgb_img)
    
    # Escalar según el factor proporcionado
    new_width = int(card['width'] * scale_factor)
    new_height = int(card['height'] * scale_factor)
    pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
    alpha = alpha.resize((new_width, new_height), Image.LANCZOS)
    
    # Rotación aleatoria
    angle = random.uniform(0, 360)
    new_width, new_height = pil_img.size
    pil_img = pil_img.rotate(angle, expand=True, resample=Image.BICUBIC)
    alpha   = alpha  .rotate(angle, expand=True, resample=Image.BICUBIC)
    dx = (pil_img.width  - new_width ) / 2
    dy = (pil_img.height - new_height) / 2
    
    img_np  = np.array(pil_img)
    alpha_np= np.array(alpha)
    
    # Ajustes de iluminación/contraste aleatorios
    brightness_factor = random.uniform(0.9, 1.1)
    contrast_factor = random.uniform(0.9, 1.1)
    saturation_factor = random.uniform(0.9, 1.1)
    if random.random() < 0.5:
        brightness_factor = random.uniform(0.3, 1.8)
    if random.random() < 0.5:
        contrast_factor = random.uniform(0.3, 1.8)
    if random.random() < 0.5:
        saturation_factor = random.uniform(0.3, 1.8)
    
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness_factor)
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast_factor)
    
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(saturation_factor)
    
    # Convertir de nuevo a OpenCV
    img_np = np.array(pil_img)
    alpha_np = np.array(alpha)
    
    # Reorganizar canales de RGB a BGR para OpenCV
    r, g, b = cv2.split(img_np)
    transformed_img = cv2.merge([b, g, r, alpha_np])
    
    # Preparar el contenedor para las bounding boxes transformadas
    transformed_bboxes = []
    
    # Matriz de transformación para la escala y rotación
    original_height, original_width = card['image'].shape[:2]
    center_x, center_y = original_width / 2, original_height / 2
    
    # Matriz de escala
    scale_matrix = np.float32([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, 1]
    ])
    
    # Matriz de rotación
    rotation_matrix = cv2.getRotationMatrix2D((center_x * scale_factor, center_y * scale_factor), angle, 1.0)
    rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])
    
    # Matriz combinada para escala y rotación
    combined_matrix = np.matmul(rotation_matrix, scale_matrix)
    
    # Transformar cada bounding box
    for bbox in card['bounding_boxes']:
        # Puntos de la bounding box original
        points = [
            [bbox['xmin'], bbox['ymin'], 1],
            [bbox['xmax'], bbox['ymin'], 1],
            [bbox['xmin'], bbox['ymax'], 1],
            [bbox['xmax'], bbox['ymax'], 1]
        ]
        
        # Transformar cada punto
        transformed_points = []
        for point in points:
            # Aplicar transformación combinada
            transformed_point = np.matmul(combined_matrix, np.array(point))
            x, y = transformed_point[:2]
            x += dx
            y += dy
            transformed_points.append([x, y])
        
        # Encontrar los límites de la nueva bounding box
        x_coords = [p[0] for p in transformed_points]
        y_coords = [p[1] for p in transformed_points]
        
        # Crear nueva bounding box
        transformed_bbox = bbox.copy()
        transformed_bbox['xmin'] = min(x_coords)
        transformed_bbox['ymin'] = min(y_coords)
        transformed_bbox['xmax'] = max(x_coords)
        transformed_bbox['ymax'] = max(y_coords)
        transformed_bbox['width'] = transformed_bbox['xmax'] - transformed_bbox['xmin']
        transformed_bbox['height'] = transformed_bbox['ymax'] - transformed_bbox['ymin']
        
        transformed_bboxes.append(transformed_bbox)
    
    # Aplicar transformación de perspectiva con probabilidad del 50%
    perspective_matrix = None
    if random.random() < 0.5:
        # Nivel de intensidad aleatorio entre 0.05 y 0.5
        perspective_intensity = random.uniform(0.05, 0.5)
        transformed_img, perspective_matrix = apply_perspective_transform(transformed_img, perspective_intensity)
        
        # Si se aplicó perspectiva, transformar las bounding boxes
        if perspective_matrix is not None:
            perspective_bboxes = []
            for bbox in transformed_bboxes:
                # Puntos de la bounding box
                points = [
                    [bbox['xmin'], bbox['ymin'], 1],
                    [bbox['xmax'], bbox['ymin'], 1],
                    [bbox['xmin'], bbox['ymax'], 1],
                    [bbox['xmax'], bbox['ymax'], 1]
                ]
                
                # Transformar cada punto
                perspective_points = []
                for point in points:
                    # Convertir a coordenadas homogéneas y aplicar transformación
                    transformed_point = np.matmul(perspective_matrix, np.array(point))
                    # Normalizar dividiendo por el último componente
                    transformed_point = transformed_point / transformed_point[2] if transformed_point[2] != 0 else transformed_point
                    perspective_points.append([transformed_point[0], transformed_point[1]])
                
                # Encontrar los límites de la nueva bounding box
                x_coords = [p[0] for p in perspective_points]
                y_coords = [p[1] for p in perspective_points]
                
                # Crear nueva bounding box
                perspective_bbox = bbox.copy()
                perspective_bbox['xmin'] = min(x_coords)
                perspective_bbox['ymin'] = min(y_coords)
                perspective_bbox['xmax'] = max(x_coords)
                perspective_bbox['ymax'] = max(y_coords)
                perspective_bbox['width'] = perspective_bbox['xmax'] - perspective_bbox['xmin']
                perspective_bbox['height'] = perspective_bbox['ymax'] - perspective_bbox['ymin']
                
                perspective_bboxes.append(perspective_bbox)
            
            transformed_bboxes = perspective_bboxes
    
    # Actualizar dimensiones
    new_card = card.copy()
    new_card['image'] = transformed_img
    new_card['width'] = transformed_img.shape[1]
    new_card['height'] = transformed_img.shape[0]
    new_card['original_angle'] = angle
    new_card['bounding_boxes'] = transformed_bboxes
    
    return new_card

def calculate_bounding_box(transformed_card, x_offset, y_offset):
    """
    Calcula las bounding boxes de una carta ya transformada en su posición final.
    
    Args:
        transformed_card: Diccionario con la información de la carta transformada
        x_offset: Desplazamiento en x de la carta en el fondo
        y_offset: Desplazamiento en y de la carta en el fondo
        
    Returns:
        Lista de diccionarios con las bounding boxes ajustadas
    """
    adjusted_boxes = []
    
    for bbox in transformed_card['bounding_boxes']:
        # Ajustar coordenadas con los desplazamientos
        adjusted_box = bbox.copy()
        adjusted_box['xmin'] = bbox['xmin'] + x_offset
        adjusted_box['ymin'] = bbox['ymin'] + y_offset
        adjusted_box['xmax'] = bbox['xmax'] + x_offset
        adjusted_box['ymax'] = bbox['ymax'] + y_offset
        
        adjusted_boxes.append(adjusted_box)
    
    return adjusted_boxes