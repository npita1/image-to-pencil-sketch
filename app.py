from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Funkcija za dodavanje paddinga slikama
def make_dimensions_divisible(image, factor=4):
    h, w, c = image.shape

    def is_power_of_2(x):
        return (x & (x - 1)) == 0

    if not is_power_of_2(h) or not is_power_of_2(w):
        new_h = h + (factor - h % factor) if h % factor != 0 else h
        new_w = w + (factor - w % factor) if w % factor != 0 else w
        padded_image = cv2.copyMakeBorder(image, 0, new_h - h, 0, new_w - w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image, h, w
    else:
        return image, h, w


# Definisanje prilagođene funkcije gubitka
@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_pred)
    return 0.5 * ssim_loss + 0.5 * mse_loss


# Učitavanje modela
model = tf.keras.models.load_model('./spaseni_modeli/model9.keras', custom_objects={'combined_loss': combined_loss})


@app.route('/generate-sketch', methods=['POST'])
def generate_sketch():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Priprema slike za model
    padded_image, original_h, original_w = make_dimensions_divisible(image, factor=4)
    padded_image = (padded_image - padded_image.min()) / (padded_image.max() - padded_image.min())
    image_input = np.expand_dims(padded_image, axis=0)

    # Generisanje skice pomoću modela
    res = model.predict(image_input)
    res = np.squeeze(res)
    res = (res - res.min()) / (res.max() - res.min())
    res = (res * 255).astype(np.uint8)

    # Uklanjanje paddinga
    res = res[:original_h, :original_w]

    # Konverzija rezultata u sliku
    sketch = Image.fromarray(res)

    img_io = io.BytesIO()
    sketch.save(img_io, format='PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
