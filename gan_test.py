import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def make_dimensions_divisible(image, factor=4):
    h, w, c = image.shape
    new_h = h + (factor - h % factor) if h % factor != 0 else h
    new_w = w + (factor - w % factor) if w % factor != 0 else w
    padded_image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image, h, w

# Učitavanje generatora (umesto celog GAN modela)
generator = tf.keras.models.load_model('./spaseni_modeli/gan_generator.keras')

# Učitavanje slike
image = cv2.imread('./dataset/woman_cartoon.jpg')

# Dodavanje paddinga (ako je potrebno)
padded_image, original_h, original_w = make_dimensions_divisible(image, factor=4)

# Skaliranje slike na 128x128 pre nego što je proslediš modelu
image_resized = cv2.resize(padded_image, (128, 128))

# Konverzija u raspon [0, 1]
image_resized = image_resized.astype(np.float32) / 255.0

# Dodavanje batch dimenzije
image_resized = np.expand_dims(image_resized, axis=0)

# Predikcija koristeći generator
res = generator.predict(image_resized)

# Uklanjanje batch dimenzije
res = np.squeeze(res)

# **Provera dimenzija izlaza**
if len(res.shape) == 3 and res.shape[-1] == 3:
    # Ako model vraća 3 kanala, konvertuj u grayscale
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

# Normalizacija slike
res = (res - res.min()) / (res.max() - res.min())

# Skaliranje na uint8 za prikaz
res = (res * 255).astype(np.uint8)

# Uklanjanje paddinga
res = res[:original_h, :original_w]

# Prikaz rezultata
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(res, cmap='gray')  # Obezbeđuje prikaz u grayscale formatu
plt.title("Generisana Skica")
plt.axis('off')

plt.show()
