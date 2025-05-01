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

def check_overlap(new_boxes, existing_boxes, max_coverage_ratio=0.4):
    """
    Verifica si alguna de las nuevas bounding boxes tapa más de un porcentaje
    máximo de cualquier caja ya existente.

    Args:
        new_boxes: lista de tuplas (xmin,ymin,xmax,ymax) de las cajas nuevas
        existing_boxes: lista de tuplas (xmin,ymin,xmax,ymax) de cajas existentes
        max_coverage_ratio: proporción máxima de cobertura permitida (p.ej. 0.4)

    Returns:
        True si alguna caja existente queda tapada > max_coverage_ratio, 
        False en caso contrario.
    """
    def area(box):
        xmin, ymin, xmax, ymax = box
        return max(0, xmax-xmin) * max(0, ymax-ymin)

    for eb in existing_boxes:
        eb_area = area(eb)
        if eb_area <= 0:
            continue
        # calculamos cobertura total de eb por todas las new_boxes
        covered = 0
        for nb in new_boxes:
            # intersección
            xi_min = max(eb[0], nb[0])
            yi_min = max(eb[1], nb[1])
            xi_max = min(eb[2], nb[2])
            yi_max = min(eb[3], nb[3])
            if xi_min < xi_max and yi_min < yi_max:
                inter = (xi_max - xi_min)*(yi_max - yi_min)
                covered += inter
        # si la cobertura total (suma de intersecciones) supera el umbral
        if covered / eb_area > max_coverage_ratio:
            return True

    return False

def check_image_coverage(image_box, existing_boxes, max_coverage_ratio=0.4):
    """
    Verifica que la caja completa `image_box` no cubra más del umbral
    de ninguna caja ya existente.

    Args:
        image_box: tupla (xmin,ymin,xmax,ymax) de la imagen entera
        existing_boxes: lista de tuplas (xmin,ymin,xmax,ymax)
        max_coverage_ratio: proporción máxima permitida

    Returns:
        True si alguna existing_box queda cubierta > max_coverage_ratio
    """
    def area(box):
        xmin, ymin, xmax, ymax = box
        return max(0, xmax-xmin)*max(0, ymax-ymin)

    xi1, yi1, xi2, yi2 = image_box
    for eb in existing_boxes:
        ex1, ey1, ex2, ey2 = eb
        # intersección
        xa = max(xi1, ex1); ya = max(yi1, ey1)
        xb = min(xi2, ex2); yb = min(yi2, ey2)
        if xa < xb and ya < yb:
            inter = (xb-xa)*(yb-ya)
            if inter / area(eb) > max_coverage_ratio:
                return True
    return False
