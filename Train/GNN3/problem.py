import psutil

# Obtenir la liste des descripteurs de fichiers ouverts par le processus actuel
file_handles = psutil.Process().open_files()

# Compter le nombre de fichiers ouverts
num_open_files = len(file_handles)

# Afficher le nombre de fichiers ouverts
print(f"Nombre de fichiers ouverts : {num_open_files}")