import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load model
MODEL_PATH = "model/resnet_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (must match your dataset)
class_labels = ["cataract", "early_cataract", "no_cataract"]

# Preprocess function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # ✅ match training preprocessing
    return img_array

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def predict_page():
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess and predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        # Customize result and caution based on prediction
        if predicted_class == "cataract":
            result = "Prediction: CATARACT"
            caution = "⚠️ Severe cataract detected. Please consult an ophthalmologist immediately for proper treatment."
        elif predicted_class == "early_cataract":
            result = "Prediction: EARLY CATARACT"
            caution = "⚠️ Early signs of cataract detected. Schedule an appointment with your eye doctor for monitoring and early intervention."
        else:  # no_cataract
            result = "Prediction: NO CATARACT"
            caution = "✅ No cataract detected. However, regular eye check-ups are recommended to maintain eye health."

        return render_template("result.html", result=result, img_path=file_path, caution=caution)

if __name__ == "__main__":
    app.run(debug=True)