import os
import numpy as np
import tifffile
from cellpose.io import logger_setup
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
from skimage import io  # Importa io desde skimage, no desde la biblioteca estándar de Python
from cellpose import models
from cellpose import io
from cellpose import metrics
from PIL import Image

def process_and_save_images(stack3D, model, folder_path):
    """
    Procesa un stack 3D con un modelo de Cellpose y guarda los resultados.

    :param stack3D: Array 3D de imágenes.
    :param model: Modelo Cellpose preentrenado.
    :param folder_path: Ruta de la carpeta donde se guardarán los resultados.
    """
    # Procesar cada imagen en el stack 3D
    for i, img in enumerate(stack3D):
        masks, flows, styles, diams = model.eval(stack3D, batch_size=8, channels=[0,0], channel_axis=None, z_axis=None, normalize=True, invert=False, rescale=None, diameter=None, do_3D=True, anisotropy=None, net_avg=False, augment=False, tile=True, tile_overlap=0.1, resample=True, interp=True, flow_threshold=0.4, cellprob_threshold=0.0, compute_masks=True, min_size=15, stitch_threshold=0.0, progress=None, loop_run=False, model_loaded=False)
        
        # Construir nombre de archivo basado en índice
        filename = os.path.join(folder_path, f"processed_image_{i}")

        # Guardar resultados para cargar en GUI
        io.masks_flows_to_seg(img, masks, flows, diams, filename, [0, 0])

        # Guardar resultados como PNG
        io.save_to_png(img, masks, flows, filename)

def create_3d_stack_from_tiff(folder_path):
    """
    Crea un stack 3D a partir de imágenes TIFF en una carpeta.

    :param folder_path: Ruta de la carpeta que contiene imágenes TIFF.
    :return: Array de NumPy con forma (nplanes, nY, nX), siendo cada 'plano' una imagen 2D.
    """
    image_files = [f for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('.tif', '.tiff'))]
    if not image_files:
        raise ValueError("No se encontraron archivos TIFF en la ruta proporcionada.")

    # Cargar la primera imagen para determinar las dimensiones
    first_image_path = os.path.join(folder_path, image_files[0])
    first_image = io.imread(first_image_path)
    nY, nX = first_image.shape

    # Inicializar un array vacío para el stack 3D
    stack3D = np.empty((len(image_files), nY, nX), dtype=first_image.dtype)

    # Llenar el stack con las imágenes
    stack3D[0] = first_image
    for i, file_name in enumerate(image_files[1:], 1):
        image_path = os.path.join(folder_path, file_name)
        stack3D[i] = io.imread(image_path)

    return stack3D

def load_mrc_as_volume(fname, mmap=False, swapxz=False):
    """
    Carga un archivo MRC y devuelve un volumen 3D como un array de NumPy.
    """
    with mrcfile.open(fname, permissive=True) as mrc:
        if mmap:
            data = mrcfile.mmap(fname, permissive=True).data
        else:
            data = mrc.data
        if swapxz:
            data = np.swapaxes(data, 0, 2)

        # Asumiendo que data es un stack de imágenes 2D
        # Apilar las imágenes para formar un volumen 3D
        volume_3d = np.stack([data[i, :, :] for i in range(data.shape[0])], axis=0)
        return volume_3d

'''
Una funcion que convertia las imagenes de .mrc a .tif pero quiero guardarlo en memoria es decir no quiero crear ficheros fisicos guardarlo en una carpeta.
'''
def convert_mrc_to_tif(images_mrc_path, output_folder):
    """
    Convierte un archivo MRC a una serie de imágenes TIFF y las guarda en una carpeta especificada.

    :param images_mrc_path: Ruta al archivo MRC.
    :param output_folder: Ruta de la carpeta donde se guardarán las imágenes TIFF.
    """
    # Crea la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with mrcfile.open(images_mrc_path, permissive=True) as mrc:
        data = mrc.data
        # Asumiendo que data es un stack de imágenes 2D
        for i in range(data.shape[0]):
            image = data[i, :, :]
            tiff_path = os.path.join(output_folder, f"image_{i}.tif")
            tifffile.imwrite(tiff_path, image)

    return output_folder


def load_mrc_as_list(fname, mmap=False, swapxz=False):
    """
    Carga un archivo MRC que contiene múltiples imágenes 2D y devuelve una lista de arrays de NumPy.
    """
    with mrcfile.open(fname, permissive=True) as mrc:
        if mmap:
            data = mrcfile.mmap(fname, permissive=True).data
        else:
            data = mrc.data
        if swapxz:
            data = np.swapaxes(data, 0, 2)

        # Asumiendo que data es un stack de imágenes 2D
        image_list = [data[i, :, :] for i in range(data.shape[0])]
        return image_list


def load_images(folder_path):
    """
    Carga todas las imágenes (TIFF, PNG, JPEG, etc.) de un directorio 
    y las devuelve como una lista de arrays de NumPy.
    """
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".tiff", ".tif", ".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            images.append(img)
    return images  # Devuelve una lista de arrays de NumPy

logger_setup()

path_tiff = r''
mask_file_path = r''
imgs = load_images(path_tiff)
masks_true = load_images(mask_file_path)
path_pretrained_model = r''
file_names= io.get_image_files(path_tiff, mask_filter="_masks")

model= models.CellposeModel(gpu=True, pretrained_model=path_pretrained_model, model_type=None, net_avg=False, diam_mean=42.0, device=None, residual_on=True, style_on=True, concatenation=False, nchan=2)

masks, flows, styles = model.eval(imgs, batch_size=15, channels=[0,0], channel_axis=None, z_axis=None, normalize=True, invert=False, rescale=None, diameter=42.0, do_3D=False, anisotropy=None, net_avg=False, augment=False, tile=True, tile_overlap=0.1, resample=True, interp=True, flow_threshold=0.7, cellprob_threshold=-1.0, compute_masks=True, min_size=15, stitch_threshold=0.0, progress=1, loop_run=False, model_loaded=False)
aij = metrics.aggregated_jaccard_index(masks_true, masks)
print("AIJ -->",aij)
ap, tp, fp, fn = metrics.average_precision(masks_true, masks, threshold=[0.5, 0.75, 0.9])
print("AP -->",ap)
#io.save_masks(imgs, masks, flows, file_names, png=True, tif=False, channels=[0, 0], suffix='', save_flows=False, save_outlines=False, save_ncolor=False, dir_above=False, in_folders=False, savedir=None, save_txt=False)
