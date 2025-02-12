import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def make_dimensions_divisible(image, factor=4):
    h, w, c = image.shape

    # Funkcija koja proverava da li su dimenzije već deljive sa 'factor'
    def is_divisible(x):
        return x % factor == 0

    # Ako dimenzije nisu deljive sa 'factor', dodaj padding
    if not is_divisible(h) or not is_divisible(w):
        new_h = h + (factor - h % factor) if h % factor != 0 else h
        new_w = w + (factor - w % factor) if w % factor != 0 else w
        padded_image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image, h, w  # Vraćamo i originalne dimenzije
    else:
        return image, h, w  # Ako su dimenzije već deljive sa 'factor', nema paddinga

# Učitavanje generatora
generator = tf.keras.models.load_model('./spaseni_modeli/gan_generator3.keras')

# Učitavanje slike
image = cv2.imread('./dataset/test_slika.jpg')

# Dodavanje paddinga (ako je potrebno) i čuvanje originalnih dimenzija
padded_image, original_h, original_w = make_dimensions_divisible(image, factor=4)

# Konverzija slike u [0, 1] raspon
padded_image = padded_image.astype(np.float32) / 255.0

# Dodavanje batch dimenzije
image_resized = np.expand_dims(padded_image, axis=0)

# Predikcija koristeći generator
res = generator.predict(image_resized)

# Uklanjanje batch dimenzije
res = np.squeeze(res)

# Provera dimenzija izlaza
print(f"Original dimensions: {original_h}x{original_w}")
print(f"Generated sketch dimensions: {res.shape}")

# Ako model vraća 3 kanala, konvertuj u grayscale
if len(res.shape) == 3 and res.shape[-1] == 3:
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

# Normalizacija slike
res = (res - res.min()) / (res.max() - res.min())

# Skaliranje na uint8 za prikaz
res = (res * 255).astype(np.uint8)

# Uklanjanje paddinga sa predikcije (ako je bilo)
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
