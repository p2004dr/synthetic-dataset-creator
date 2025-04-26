"""
Utilidades para la generación y manejo de anotaciones en formato YOLO.
"""

def convert_to_yolo_format(box, image_width, image_height, class_id):
    """
    Convierte una bounding box de formato (xmin, ymin, xmax, ymax) a formato YOLO 
    (x_center, y_center, width, height). Todos los valores YOLO están normalizados entre 0 y 1.
    
    Args:
        box: Tupla (xmin, ymin, xmax, ymax) de la caja
        image_width: Ancho de la imagen
        image_height: Alto de la imagen
        class_id: ID de la clase
        
    Returns:
        Tupla (class_id, x_center, y_center, width, height)
    """
    xmin, ymin, xmax, ymax = box
    
    # Normalizar entre 0 y 1
    x_center = (xmin + xmax) / (2 * image_width)
    y_center = (ymin + ymax) / (2 * image_height)
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    
    # Asegurarse de que los valores estén dentro del rango [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return class_id, x_center, y_center, width, height

def create_yolo_annotation(objects, image_width, image_height, label_to_id):
    """
    Crea una anotación en formato YOLO.
    
    Args:
        objects: Lista de objetos con 'label', 'xmin', 'ymin', 'xmax', 'ymax'
        image_width: Ancho de la imagen
        image_height: Alto de la imagen
        label_to_id: Diccionario que mapea etiquetas a IDs de clase
        
    Returns:
        Lista de líneas de anotación en formato YOLO
    """
    annotation_lines = []
    
    for obj in objects:
        label = obj['label']
        class_id = label_to_id.get(label, -1)
        
        if class_id == -1:
            print(f"[WARN] La etiqueta '{label}' no está en la lista de clases. Ignorando.")
            continue
        
        # Convertir coordenadas a formato YOLO
        yolo_box = convert_to_yolo_format(
            (obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']),
            image_width, image_height, class_id
        )
        
        # Añadir línea al archivo de anotación
        line = f"{yolo_box[0]} {yolo_box[1]:.5f} {yolo_box[2]:.5f} {yolo_box[3]:.5f} {yolo_box[4]:.5f}"
        annotation_lines.append(line)
    
    return annotation_lines

def save_yolo_annotation(annotation_lines, output_path):
    """
    Guarda las líneas de anotación en un archivo.
    
    Args:
        annotation_lines: Lista de líneas de anotación
        output_path: Ruta donde guardar el archivo
    """
    with open(output_path, 'w') as label_file:
        for line in annotation_lines:
            label_file.write(line + '\n')

def check_overlap(box1, boxes, max_overlap_ratio=0.5, max_coverage_ratio=0.8):
    """
    Verifica si una nueva bounding box se solapa demasiado con las existentes.
    
    Args:
        box1: Tupla (xmin, ymin, xmax, ymax) de la nueva caja
        boxes: Lista de tuplas (xmin, ymin, xmax, ymax) de cajas existentes
        max_overlap_ratio: Proporción máxima de solapamiento permitida
        max_coverage_ratio: Proporción máxima de cobertura permitida
        
    Returns:
        True si hay solapamiento excesivo, False en caso contrario
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    
    if area1 <= 0:
        return True  # Caja inválida
    
    for box2 in boxes:
        xmin2, ymin2, xmax2, ymax2 = box2
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        
        # Calcular el área de intersección
        xmin_intersect = max(xmin1, xmin2)
        ymin_intersect = max(ymin1, ymin2)
        xmax_intersect = min(xmax1, xmax2)
        ymax_intersect = min(ymax1, ymax2)
        
        # Si hay intersección
        if xmin_intersect < xmax_intersect and ymin_intersect < ymax_intersect:
            intersection_area = (xmax_intersect - xmin_intersect) * (ymax_intersect - ymin_intersect)
            
            # Calcular ratios de solapamiento respecto a ambas áreas
            overlap_ratio1 = intersection_area / area1  # Cuánto del área de la nueva caja está solapada
            overlap_ratio2 = intersection_area / area2  # Cuánto del área de la caja existente está solapada
            
            # Rechazar si:
            # 1. La nueva carta está demasiado solapada con una existente
            # 2. La nueva carta cubre demasiado de una existente
            if overlap_ratio1 > max_overlap_ratio or overlap_ratio2 > max_overlap_ratio:
                return True
                
            # 3. Verificar si una carta está mayormente cubierta por otra (más del max_coverage_ratio)
            if overlap_ratio1 > max_coverage_ratio or overlap_ratio2 > max_coverage_ratio:
                return True
    
    return False  # No hay solapamiento excesivo