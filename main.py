# import os
# import cv2
#
# normal_dir = './dataset/original'
# sketch_dir = './dataset/sketch'
#
# print("Broj originalnih slika:", len(os.listdir(normal_dir)))
# print("Broj skica:", len(os.listdir(sketch_dir)))
#
# img_sample = cv2.imread(os.path.join(normal_dir, os.listdir(normal_dir)[0]))
# print("Dimenzije uzorka slike:", img_sample.shape)
#
# import tensorflow as tf
#
# print("Dostupni uređaji:", tf.config.list_physical_devices())

import os
import shutil
import random


def move_images(source_dir, target_dir, total_images=3000):
    # Dobijamo listu svih foldera unutar source_dir
    all_folders = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]

    # Lista za prikupljanje slika
    all_images = []

    # Uzimamo slike iz svakog foldera
    for folder in all_folders:
        folder_path = os.path.join(source_dir, folder)
        images_in_folder = [img for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

        # Dodajemo slike iz trenutnog foldera u listu
        for img in images_in_folder:
            all_images.append(os.path.join(folder_path, img))

    # Randomizujemo listu slika da bismo ih uzeli nasumično
    random.shuffle(all_images)

    # Uzimamo tačno 3000 slika, ili manje ako nemamo dovoljno
    selected_images = all_images[:total_images]

    # Premještamo slike u ciljani folder
    for i, img_path in enumerate(selected_images):
        target_image_path = os.path.join(target_dir, f'image_{i + 1}.jpg')
        shutil.copy(img_path, target_image_path)
        print(f'Premeštena slika: {img_path} -> {target_image_path}')


# Glavni folder sa podfolderima
source_directory = '../../../sketchy dataset'

# Ciljani folder za premještanje slika
target_directory = '../../../dataset256/original'

# Proveri da ciljani folder postoji, ako ne, napravi ga
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Pokreni funkciju za premeštanje slika
move_images(source_directory, target_directory, total_images=3000)

