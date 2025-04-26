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
        Imagen transformada y puntos de esquina transformados
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
    
    # Retornar los puntos transformados junto con la imagen para calcular bounding box
    transformed_corners = []
    for point in pts1:
        # Convertir cada punto usando la matriz de transformación
        # Necesitamos agregar 1 para hacer la transformación homogénea
        point_homogeneous = np.array([[point[0]], [point[1]], [1]])
        transformed_point = M @ point_homogeneous
        # Normalizar dividiendo por el último componente
        transformed_point = transformed_point / transformed_point[2]
        transformed_corners.append((transformed_point[0][0], transformed_point[1][0]))
    
    return warped_img, transformed_corners

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
    pil_img = pil_img.rotate(angle, expand=True, resample=Image.BICUBIC)
    alpha = alpha.rotate(angle, expand=True, resample=Image.BICUBIC)
    
    # Ajustes de iluminación/contraste aleatorios
    brightness_factor = random.uniform(0.9, 1.1)
    contrast_factor = random.uniform(0.9, 1.1)
    saturation_factor = random.uniform(0.9, 1.1)
    if random.random() < 0.5:
        brightness_factor = random.uniform(0.3, 1.7)
    if random.random() < 0.5:
        contrast_factor = random.uniform(0.3, 1.7)
    if random.random() < 0.5:
        saturation_factor = random.uniform(0.3, 1.7)
    
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
    
    # Aplicar transformación de perspectiva con probabilidad del 50%
    if random.random() < 0.5:
        # Nivel de intensidad aleatorio entre 0.05 y 0.7
        perspective_intensity = random.uniform(0.05, 0.5)
        transformed_img, _ = apply_perspective_transform(transformed_img, perspective_intensity)
    
    # Actualizar dimensiones
    new_card = card.copy()
    new_card['image'] = transformed_img
    new_card['width'] = transformed_img.shape[1]
    new_card['height'] = transformed_img.shape[0]
    new_card['original_angle'] = angle
    
    return new_card

def calculate_bounding_box(card, x_offset, y_offset):
    """
    Calcula la bounding box de la carta después de transformaciones.
    Considera los píxeles no transparentes para generar una bounding box precisa.
    
    Args:
        card: Diccionario con información de la carta
        x_offset: Desplazamiento en X
        y_offset: Desplazamiento en Y
        
    Returns:
        Tupla (xmin, ymin, xmax, ymax)
    """
    img = card['image']
    height, width = img.shape[:2]
    
    # Encontrar todos los píxeles no transparentes (alpha > 0)
    alpha_channel = img[:, :, 3]
    non_transparent_indices = np.where(alpha_channel > 0)
    
    if len(non_transparent_indices[0]) > 0 and len(non_transparent_indices[1]) > 0:
        # Calcular las coordenadas mínimas y máximas de los píxeles no transparentes
        ymin = np.min(non_transparent_indices[0])
        ymax = np.max(non_transparent_indices[0])
        xmin = np.min(non_transparent_indices[1])
        xmax = np.max(non_transparent_indices[1])
        
        # Aplicar offset
        xmin += x_offset
        xmax += x_offset
        ymin += y_offset
        ymax += y_offset
        
        # Asegurarse de que están dentro de los límites
        return int(xmin), int(ymin), int(xmax), int(ymax)
    else:
        # Si no hay píxeles no transparentes, devolver un rectángulo predeterminado
        return x_offset, y_offset, x_offset + width, y_offset + height