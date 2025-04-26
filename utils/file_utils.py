"""
Utilidades para manejo de archivos y directorios.
"""
import os
from datetime import datetime
import uuid

def create_directory_structure(directories):
    """
    Crea la estructura de directorios necesaria para el dataset.
    
    Args:
        directories: Lista de directorios a crear
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_data_yaml(output_dir, classes):
    """
    Crea el archivo data.yaml en el directorio del dataset según el formato solicitado.
    
    Args:
        output_dir: Directorio de salida del dataset
        classes: Lista de nombres de clases
    """
    yaml_content = f"""train: ./train/images
val: ./valid/images
nc: {len(classes)}
names: {classes}
"""
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)
    
    print(f"Archivo data.yaml creado en {yaml_path}")

def print_directory_structure():
    """Imprime la estructura de directorios del dataset generado."""
    print("dataset/")
    print("├── data.yaml")
    print("├── train/")
    print("│   ├── images/  (con archivos .jpg, .jpeg o .png)")
    print("│   └── labels/  (con archivos .txt)")
    print("├── valid/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("└── test/")
    print("    ├── images/")
    print("    └── labels/")

def generate_unique_filename(prefix="IMG", extension=".jpg"):
    """
    Genera un nombre de archivo único basado en la fecha y un UUID.
    
    Args:
        prefix: Prefijo para el nombre de archivo
        extension: Extensión del archivo
        
    Returns:
        Nombre de archivo único
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}{timestamp}_{uuid.uuid4().hex[:8]}{extension}"