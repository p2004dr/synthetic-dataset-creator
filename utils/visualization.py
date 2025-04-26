"""
Funciones para visualización de datasets y anotaciones.
"""
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset_samples(dataset_type, images_dir, labels_dir, classes, num_samples=3):
    """
    Visualiza muestras del dataset con sus bounding boxes en formato YOLO.
    
    Args:
        dataset_type: Nombre del conjunto de datos (train, valid, test)
        images_dir: Directorio con las imágenes
        labels_dir: Directorio con las etiquetas
        classes: Lista de nombres de clases
        num_samples: Número de muestras a visualizar
    """
    print(f"\nVisualizando muestras de {dataset_type}...")
    
    # Elegir aleatoriamente algunas muestras
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    if len(label_files) > num_samples:
        selected_files = random.sample(label_files, num_samples)
    else:
        selected_files = label_files
    
    plt.figure(figsize=(15, 5*num_samples))
    
    for i, label_file in enumerate(selected_files):
        # Construir la ruta de la imagen correspondiente
        image_file = os.path.splitext(label_file)[0] + '.jpg'
        image_path = os.path.join(images_dir, image_file)
        
        # Verificar si la imagen existe
        if not os.path.exists(image_path):
            print(f"Advertencia: No se encontró la imagen {image_file}")
            continue
        
        # Cargar imagen
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # Leer el archivo de etiquetas YOLO
        boxes = []
        class_ids = []
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f.readlines():
                if line.strip():
                    values = line.strip().split()
                    if len(values) == 5:
                        class_id = int(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        
                        # Convertir de formato YOLO a coordenadas de píxeles
                        xmin = int((x_center - width/2) * img_width)
                        ymin = int((y_center - height/2) * img_height)
                        xmax = int((x_center + width/2) * img_width)
                        ymax = int((y_center + height/2) * img_height)
                        
                        boxes.append((xmin, ymin, xmax, ymax))
                        class_ids.append(class_id)
        
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(image)
        plt.title(f"{dataset_type} - {image_file}")
        plt.axis('off')
        
        # Dibujar cajas delimitadoras
        for (xmin, ymin, xmax, ymax), class_id in zip(boxes, class_ids):
            # Obtener nombre de clase
            class_name = classes[class_id]
            
            # Dibujar rectángulo
            rect = plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                fill=False, edgecolor='cyan', linewidth=2
            )
            plt.gca().add_patch(rect)
            
            # Añadir etiqueta
            plt.text(
                xmin, ymin - 5, class_name, 
                bbox=dict(facecolor='cyan', alpha=0.7), 
                fontsize=10, color='black'
            )
    
    plt.tight_layout()
    plt.show()