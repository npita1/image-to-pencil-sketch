import numpy as np
import cv2
import os

normal = os.listdir('./dataset/original')
sketched = os.listdir('./dataset/sketch')
normal = sorted(normal)
sketched = sorted(sketched)

X = []
Y = []

for i in range(0, 3000):
    f = normal[i]
    img = cv2.imread(os.path.join('./dataset/original', f))
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = (img - img.min()) / (img.max() - img.min())
    X.append(img)
    f = sketched[i]
    img = cv2.imread(os.path.join('./dataset/sketch', f))
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = (img - img.min()) / (img.max() - img.min())
    Y.append(img)

X = np.array(X)
Y = np.array(Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

import tensorflow as tf

# Arhitektura modela
input = tf.keras.layers.Input([128, 128, 3])
output = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input)
output = tf.keras.layers.MaxPooling2D((2, 2))(output)
output = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(output)
output = tf.keras.layers.MaxPooling2D((2, 2))(output)
output = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(output)
output = tf.keras.layers.UpSampling2D((2, 2))(output)
output = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(output)
output = tf.keras.layers.UpSampling2D((2, 2))(output)
output = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(output)

model = tf.keras.models.Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Dodavanje Early Stopping callback-a
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Metrika za praćenje
    patience=5,         # Broj epoha za čekanje prije zaustavljanja
    restore_best_weights=True  # Vraća težine modela na najbolje stanje
)

# Treniranje modela
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=700,  # Maksimalan broj epoha
    callbacks=[early_stop]  # Dodaj Early Stopping
)

# Čuvanje modela
model.save('./spaseni_modeli/model5.keras')

print("Zavrseno treniranje i spasen model.")