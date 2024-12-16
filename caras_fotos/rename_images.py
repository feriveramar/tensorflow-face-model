import os

# Directorios específicos
path_sakura = "/home/sakura/quinto/design/TF_Faces_Model/caras_fotos/sakura"
path_agustin = "/home/sakura/quinto/design/TF_Faces_Model/caras_fotos/agustin"

# Función para renombrar las imágenes
def rename_images_in_folder(folder_path, prefix):
    # Obtener lista de archivos en la carpeta
    images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # Ordenar las imágenes para renombrarlas de manera consecutiva
    images.sort()

    # Renombrar las imágenes con el prefijo adecuado
    for index, image in enumerate(images, start=1):
        old_name = os.path.join(folder_path, image)
        new_name = os.path.join(folder_path, f"{prefix}{index}.jpg")
        os.rename(old_name, new_name)

        print(f"Renombrado: {old_name} -> {new_name}")

# Renombrar imágenes dentro de la carpeta "sakura"
rename_images_in_folder(path_sakura, "saku")

# Renombrar imágenes dentro de la carpeta "agustin"
rename_images_in_folder(path_agustin, "agus")
