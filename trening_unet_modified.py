import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Putanje do foldera sa originalnim i skiciranim slikama
normal_dir = './dataset/original'
sketch_dir = './dataset/sketch'

# Dimenzije na koje će se sve slike skalirati
target_size = (128, 128)

# Učitavanje i skaliranje slika
normal = sorted(os.listdir(normal_dir))
sketched = sorted(os.listdir(sketch_dir))

X, Y = [], []

for i in range(len(normal)):
    # Učitavanje originalne slike
    f_normal = os.path.join(normal_dir, normal[i])
    img_normal = cv2.imread(f_normal)
    img_normal = cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB)
    img_normal = cv2.resize(img_normal, target_size, interpolation=cv2.INTER_AREA)

    # Učitavanje skicirane slike
    f_sketch = os.path.join(sketch_dir, sketched[i])
    img_sketch = cv2.imread(f_sketch)
    img_sketch = cv2.cvtColor(img_sketch, cv2.COLOR_BGR2RGB)
    img_sketch = cv2.resize(img_sketch, target_size, interpolation=cv2.INTER_AREA)

    # Dodavanje u liste
    X.append(img_normal)
    Y.append(img_sketch)

# Pretvaranje u numpy niz i normalizacija
X = np.array(X) / 255.0
Y = np.array(Y) / 255.0

# Podjela dataset-a na trening i test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Definisanje modela sa fleksibilnim ulaznim dimenzijama
inputs = tf.keras.layers.Input(shape=(None, None, 3))

# Encoder
c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
p1 = tf.keras.layers.Dropout(0.3)(p1)

c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
p2 = tf.keras.layers.Dropout(0.3)(p2)

# Bottleneck
b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(b)

# Decoder
u2 = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(b)
u2 = tf.keras.layers.UpSampling2D((2, 2))(u2)
u2 = tf.keras.layers.Concatenate()([u2, c2])

u1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(u2)
u1 = tf.keras.layers.UpSampling2D((2, 2))(u1)
u1 = tf.keras.layers.Concatenate()([u1, c1])

outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(u1)
def combined_loss(y_true, y_pred):
    ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)

    # Kreiranje instanciranog objekta klase MeanSquaredError
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_pred)

    return 0.5 * ssim_loss + 0.5 * mse_loss

model = tf.keras.models.Model(inputs, outputs)
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])

# Early Stopping i treniranje
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=250,
    batch_size=16,
    callbacks=[early_stop]
)

# Čuvanje modela
model.save('./spaseni_modeli/model9.keras')

print("Završeno treniranje i sačuvan model.")
