from flask import Flask, request, jsonify, render_template, url_for
import os
import logging
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

try:
    model = load_model('model.keras')
    logging.info(" Model loaded successfully")
except Exception as e:
    logging.error(" Error loading model", exc_info=True)
    model = None

try:
    with open('class_indices.json') as f:
        class_indices = json.load(f)
        index_to_label = {int(v): k for k, v in class_indices.items()}
    logging.info(" Class indices loaded")
except Exception as e:
    logging.error(" Failed to load class indices", exc_info=True)
    index_to_label = {}

target_img = os.path.join(os.getcwd(), 'static/images')

@app.route('/')
def main_index():
    return render_template('index.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not os.path.exists(target_img):
            os.makedirs(target_img)

        file_path = os.path.join(target_img, file.filename)
        file.save(file_path)

        image = load_img(file_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        butterfly_name = index_to_label.get(predicted_class, "Unknown")
        relative_image_path = url_for('static', filename=f'images/{file.filename}')

        logging.info(f" Prediction: {butterfly_name} ({confidence:.2f})")

        return render_template('output.html',
                               butterfly=butterfly_name,
                               user_image=relative_image_path,
                               confidence=f"{confidence:.2f}")
    except Exception as e:
        logging.error(" Prediction failed", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(target_img):
        os.makedirs(target_img)
    app.run(debug=True)
