import mrcfile
import numpy as np

# Leer el archivo .mrc de las imágenes
with mrcfile.open('', permissive=True) as mrc:
    vol_imagenes = mrc.data

# Leer el archivo .mrc de las máscaras
with mrcfile.open('05_organelles.mrc', permissive=True) as mrc:
    vol_mascaras = mrc.data

# Verificar que las dimensiones coincidan
assert vol_imagenes.shape == vol_mascaras.shape, "Las dimensiones de las imágenes y las máscaras no coinciden."

# Verificar que las máscaras estén binarizadas/etiquetadas correctamente
# Asumiendo que las máscaras deben tener valores enteros (0 para el fondo, 1, 2, 3,... para cada célula)

# Verificar si el fondo es 0
assert (vol_mascaras == 0).any(), "El fondo no está presente en las máscaras (debería ser 0)."

# Verificar si cada objeto tiene un identificador único y si son enteros
unique_values = np.unique(vol_mascaras)
assert all(isinstance(value, np.integer) for value in unique_values), "Los identificadores de los objetos no son todos enteros."

# Opcional: Verificar que no hay identificadores negativos o no deseados
assert all(value >= 0 for value in unique_values), "Hay identificadores negativos en las máscaras."

print("Verificación completada: Todo parece estar correcto.")
print(f"Forma del volumen: {vol_imagenes.shape}")
print(f"Tipo de datos del volumen: {vol_imagenes.dtype}")
print(f"---------------------------")

print(f"Forma del volumen: {vol_mascaras.shape}")
print(f"Tipo de datos del volumen: {vol_mascaras.dtype}")
