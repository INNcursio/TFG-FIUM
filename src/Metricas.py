import numpy as np
from skimage.io import imread
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from cellpose import metrics
  
def calculate_iou(mask_true, mask_pred):
    # Intersección
    intersection = np.logical_and(mask_true, mask_pred)
    # Unión
    union = np.logical_or(mask_true, mask_pred)
    # IoU
    iou = np.sum(intersection) / np.sum(union)
    return iou
# Carga las imágenes desde las rutas
ruta_mascara_solucion = r''
ruta_mascara_predicha = r''

mascara_solucion = imread(ruta_mascara_solucion)
mascara_predicha = imread(ruta_mascara_predicha)


# Asegúrate de que las imágenes sean binarias (solo contengan 0 y 255) y estén en formato uint8
mascara_solucion = (mascara_solucion > 0).astype(np.uint8)
mascara_predicha = (mascara_predicha > 0).astype(np.uint8)

# Calcula la precisión
precision = precision_score(mascara_solucion.flatten(), mascara_predicha.flatten())

# Calcula el recall
recall = recall_score(mascara_solucion.flatten(), mascara_predicha.flatten())

# Calcula el f1_score
f1 = f1_score(mascara_solucion.flatten(), mascara_predicha.flatten())

accuarucy_score = accuracy_score(mascara_solucion,mascara_predicha, normalize=True )

print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Acuaricy Score:", accuarucy_score)

iou = calculate_iou(mascara_solucion, mascara_predicha)
print("IoU:", iou)
#aij = metrics.aggregated_jaccard_index(mascara_solucion, mascara_predicha)
#print("AIJ -->",aij)
ap, tp, fp, fn = metrics.average_precision(mascara_solucion, mascara_predicha, threshold=[0.5, 0.75, 0.9])
print("AP -->",ap)
