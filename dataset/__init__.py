"""
Módulos para la generación del dataset sintético.
"""
__all__ = ['generate_dataset']

# Importamos después de definir __all__ para evitar importación circular
from dataset.generator import generate_dataset
