import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def make_dimensions_divisible(image, factor=4):
    h, w, c = image.shape

    # Funkcija koja proverava da li su dimenzije već potencija broja 2
    def is_power_of_2(x):
        return (x & (x - 1)) == 0

    # Ako dimenzije već nisu potencija broja 2, dodaj padding
    if not is_power_of_2(h) or not is_power_of_2(w):
        new_h = h + (factor - h % factor) if h % factor != 0 else h
        new_w = w + (factor - w % factor) if w % factor != 0 else w
        padded_image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image, h, w  # Vraćamo i originalne dimenzije
    else:
        return image, h, w  # Ako su dimenzije već potencija broja 2, nema paddinga

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)

    # Kreiranje instanciranog objekta klase MeanSquaredError
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_pred)

    return 0.5 * ssim_loss + 0.5 * mse_loss


# Učitavanje modela
#model = tensorflow.keras.models.load_model('./spaseni_modeli/model8.keras')

# Učitavanje modela 9
model = tf.keras.models.load_model('./spaseni_modeli/model9.keras', custom_objects={'combined_loss': combined_loss})

# Učitavanje slike
image = cv2.imread('./dataset/n02691156_9453.jpg')

# Dodavanje paddinga (ako je potrebno) i čuvanje originalnih dimenzija
padded_image, original_h, original_w = make_dimensions_divisible(image, factor=4)

# Normalizacija slike
padded_image = (padded_image - padded_image.min()) / (padded_image.max() - padded_image.min())

# Dodavanje batch dimenzije
image = np.expand_dims(padded_image, axis=0)

# Predikcija
res = model.predict(image)

# Uklanjanje batch dimenzije i skaliranje rezultata
res = np.squeeze(res)
res = (res - res.min()) / (res.max() - res.min())
res = (res * 255).astype(np.uint8)

# Uklanjanje paddinga sa predikcije
res = res[:original_h, :original_w]

# Skaliranje originalne slike na uint8 za prikaz
original_image = (image[0] * 255).astype(np.uint8)
original_image = original_image[:original_h, :original_w]  # Uklanjanje paddinga sa originalne slike

# Prikaz originalne slike i predikcije
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title("Skica")
plt.axis('off')

plt.show()
