import tensorflow as tf
import numpy as np




# Učitavanje modela
model = tf.keras.models.load_model('./spaseni_modeli/model9.keras')

# Evaluacija na testnom skupu
loss, accuracy = model.evaluate(X_test, Y_test, batch_size=16)

print(f'Testni gubitak (loss): {loss:.4f}')
print(f'Testna tačnost (accuracy): {accuracy:.4f}')

from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

# Predikcija na testnom skupu
predictions = model.predict(X_test)

# Računanje metrika
mae = mean_absolute_error(Y_test.flatten(), predictions.flatten())
mse = mean_squared_error(Y_test.flatten(), predictions.flatten())

# Računanje PSNR i SSIM (za procjenu kvaliteta slike)
psnr = tf.image.psnr(Y_test, predictions, max_val=1.0)
ssim = tf.image.ssim(Y_test, predictions, max_val=1.0)

print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Prosječan PSNR: {np.mean(psnr):.2f} dB')
print(f'Prosječan SSIM: {np.mean(ssim):.4f}')


import matplotlib.pyplot as plt

# Izračunaj greške
errors = np.abs(Y_test - predictions)

# Prikaz histograma grešaka
plt.hist(errors.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Distribucija grešaka')
plt.xlabel('Vrijednost greške')
plt.ylabel('Frekvencija')
plt.show()

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]

    model.fit(X_train, Y_train, epochs=50, batch_size=16, verbose=0)
    val_loss = model.evaluate(X_val, Y_val, verbose=0)

    print(f'Fold {fold}, Validation Loss: {val_loss:.4f}')
    fold += 1


model8 = load_model('./spaseni_modeli/model8.keras')
model9 = load_model('./spaseni_modeli/model9.keras')

loss8, acc8 = model8.evaluate(X_test, Y_test, batch_size=16)
loss9, acc9 = model9.evaluate(X_test, Y_test, batch_size=16)

print(f'Model 8 - Loss: {loss8:.4f}, Accuracy: {acc8:.4f}')
print(f'Model 9 - Loss: {loss9:.4f}, Accuracy: {acc9:.4f}')


