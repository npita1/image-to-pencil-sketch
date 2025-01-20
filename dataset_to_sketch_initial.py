import cv2 as cv
import os

# Postavljanje putanja
input_dir = "./dataset/original"
output_dir = "./dataset/sketch"
os.makedirs(output_dir, exist_ok=True)

# Funkcija za pretvaranje slike u skicu
def convert_to_sketch(image_path, output_path):
    # Učitavanje slike
    image = cv.imread(image_path)
    if image is None:
        print(f"Ne mogu učitati sliku: {image_path}")
        return

    # Pretvaranje u sivu sliku
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Inverzija sive slike
    invert_image = cv.bitwise_not(gray_image)

    # Zamućenje slike
    blur_image = cv.GaussianBlur(invert_image, (21, 21), 0)

    # Inverzija zamućene slike
    invert_blur = cv.bitwise_not(blur_image)

    # Kreiranje skice
    sketch = cv.divide(gray_image, invert_blur, scale=256.0)

    # Spremanje skice
    cv.imwrite(output_path, sketch)

# Obrada svih slika iz input foldera
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    output_path = os.path.join(output_dir, img_name)

    # Pretvaranje slike u skicu
    convert_to_sketch(img_path, output_path)

print("Sve slike su uspješno pretvorene u skice!")
