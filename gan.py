import numpy as np
import tensorflow as tf
import os
import cv2
from sklearn.model_selection import train_test_split

# Učitavanje dataset-a
normal_dir = './dataset/original'
sketch_dir = './dataset/sketch'

target_size = (128, 128)

X, Y = [], []

for img_name in sorted(os.listdir(normal_dir)):
    f_normal = os.path.join(normal_dir, img_name)
    f_sketch = os.path.join(sketch_dir, img_name)

    img_normal = cv2.imread(f_normal)
    img_normal = cv2.cvtColor(img_normal, cv2.COLOR_BGR2RGB)
    img_normal = cv2.resize(img_normal, target_size) / 255.0

    img_sketch = cv2.imread(f_sketch)
    img_sketch = cv2.cvtColor(img_sketch, cv2.COLOR_BGR2RGB)
    img_sketch = cv2.resize(img_sketch, target_size) / 255.0

    X.append(img_normal)
    Y.append(img_sketch)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Generator (U-Net)
def build_generator():
    inputs = tf.keras.layers.Input(shape=(None, None, 3))

    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    b = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(b)

    u2 = tf.keras.layers.UpSampling2D((2, 2))(b)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    u2 = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(u2)

    u1 = tf.keras.layers.UpSampling2D((2, 2))(u2)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    u1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(u1)

    outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid')(u1)

    return tf.keras.models.Model(inputs, outputs, name="Generator")


# Discriminator
def build_discriminator():
    inputs = tf.keras.layers.Input(shape=(None, None, 3))

    # Konvolucioni slojevi sa LeakyReLU aktivacijama
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same')(inputs)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Global pooling za podršku bilo kojim dimenzijama
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Fully connected sloj za izlaz
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs, x, name="Discriminator")


# Definisanje GAN-a
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                      metrics=['accuracy'])

discriminator.trainable = False
inputs = tf.keras.layers.Input(shape=(None, None, 3))
generated = generator(inputs)
gan_output = discriminator(generated)
GAN = tf.keras.models.Model(inputs, [generated, gan_output])
GAN.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[100, 1],
            optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5))


# Treniranje GAN-a
def train_gan(epochs=100, batch_size=16):
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        real_sketches = Y_train[idx]

        fake_sketches = generator.predict(real_images)
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_sketches, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_sketches, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = GAN.train_on_batch(real_images, [real_sketches, real_labels])

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}: D loss={d_loss[0]:.4f}, G loss={g_loss[0]:.4f}")

    #generator.save('./spaseni_modeli/gan_generator4.keras')
    print("Treniranje završeno, generator sačuvan.")


# Pokretanje treniranja
train_gan(epochs=5000, batch_size=16)