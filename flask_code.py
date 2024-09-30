import pickle

import cv2
import numpy as np
from flask import Flask, render_template, request, redirect
import numpy as np
from PIL import Image

from tensorflow.keras import models

app = Flask(__name__)

labels = pickle.load(open('model_1_labels.pkl', 'rb'))
model_1 = models.load_model('best_model_1.keras')
model_2 = models.load_model('best_model_2.keras')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'img' not in request.files:
        return render_template('index.html', prediction_text_1="No file part")

    try:
        file = request.files['img']

        if file.filename == '':
            return render_template('index.html', prediction_text_1="No selected file")

        # Load and preprocess the image
        image_array = np.array(Image.open(file.stream)) / 255.0
        image_array = cv2.resize(image_array, (224, 224))

        # Ensure the image has the correct shape (1, 224, 224, 3)
        if image_array.ndim == 2:  # Grayscale image
            image_array = np.stack((image_array,) * 3, axis=-1)  # Convert to RGB
        elif image_array.shape[2] == 4:  # RGBA image
            image_array = image_array[:, :, :3]  # Drop the alpha channel

        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict 1
        pred_1 = model_1.predict(image_array)
        pred_index_1 = np.argmax(pred_1[0])  # Get the index of the highest probability
        pred_label_1 = labels[pred_index_1]  # Map to the label

        # Predict 2
        pred_2 = model_2.predict(image_array)
        pred_index_2 = np.argmax(pred_2[0])  # Get the index of the highest probability
        pred_label_2 = labels[pred_index_2]  # Map to the label

        msg_1 = f'Prediction 1 is:{pred_label_1}'
        msg_2 = f'Prediction 2 is:{pred_label_2}'

    except Exception as e:
        msg = f'Error has occurred: {e}'
        print(msg)
    return render_template('index.html', prediction_text_1=msg_1, prediction_text_2=msg_2)


@app.route('/again', methods=['POST'])
def again():
    return render_template('index.html', prediction_text_1=None, prediction_text_2=None)


if __name__ == '__main__':
    app.run(debug=True)
