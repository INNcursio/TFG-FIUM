import os
import numpy as np
from skimage import io
from skimage.measure import label
from PIL import Image

def etiquetar_mascaras_en_carpeta(path_carpeta):
    """
    Etiqueta todas las máscaras binarias en una carpeta y las guarda o devuelve.

    :param path_carpeta: Ruta de la carpeta que contiene imágenes de máscaras binarias.
    """
    for filename in os.listdir(path_carpeta):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            # Cargar la imagen
            ruta_completa = os.path.join(path_carpeta, filename)
            mascara_binaria = io.imread(ruta_completa)
            
            # Asegurarse de que la máscara está en formato binario
            mascara_binaria = (mascara_binaria > 0).astype(int)

            # Etiquetar la máscara
            mascara_etiquetada = label(mascara_binaria)

            # Guardar o procesar la máscara etiquetada
            # Opción 1: Guardar la imagen etiquetada
            ruta_guardado = os.path.join(path_carpeta, 'etiquetadas', filename)
            os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
            Image.fromarray(mascara_etiquetada.astype(np.uint8)).save(ruta_guardado)


# Uso de la función
path=r''
etiquetar_mascaras_en_carpeta(path)
