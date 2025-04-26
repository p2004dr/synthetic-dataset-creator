"""
Punto de entrada principal para la generación del dataset sintético.
"""
import time
from config import DATASET_CONFIG, DIRECTORIES
from utils.file_utils import create_directory_structure, print_directory_structure, clear_dataset
from utils.visualization import visualize_dataset_samples
from dataset.generator import generate_dataset

def main():
    """Función principal."""
    # Usar directamente la configuración por defecto
    config = DATASET_CONFIG.copy()
    config['test_ratio'] = 1.0 - config['train_ratio'] - config['valid_ratio']

    clear_dataset(config['output_dir'])
    
    print("=== Generador de Dataset Sintético ===")
    print(f"Imágenes totales: {config['total_images']}")
    print(f"Distribución: {config['train_ratio']*100:.1f}% Train, "
          f"{config['valid_ratio']*100:.1f}% Valid, "
          f"{config['test_ratio']*100:.1f}% Test")
    
    # Crear estructura de directorios
    create_directory_structure(DIRECTORIES)
    
    # Medir tiempo de ejecución
    start_time = time.time()
    
    # Generar dataset
    total_generated = generate_dataset(config)
    
    # Mostrar tiempo de ejecución
    elapsed_time = time.time() - start_time
    print(f"\nTiempo total de generación: {elapsed_time:.2f} segundos")
    print(f"Velocidad: {total_generated/elapsed_time:.2f} imágenes/segundo")
    
    # Mostrar estructura del dataset generado
    print("\nEstructura del dataset generado:")
    print_directory_structure()
    
    # Visualizar algunas muestras (opcional, podrías establecer esto como constante en config.py)
    show_samples = False  # Cambia a False si no quieres visualizar muestras
    if show_samples:
        visualize_dataset_samples("Train Dataset", config['train_images_dir'], config['train_labels_dir'], config['classes'])
        visualize_dataset_samples("Validation Dataset", config['valid_images_dir'], config['valid_labels_dir'], config['classes'])
        visualize_dataset_samples("Test Dataset", config['test_images_dir'], config['test_labels_dir'], config['classes'])
    
    print("\n¡Generación completada!")

if __name__ == "__main__":
    main()