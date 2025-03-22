from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load your trained model (Replace 'model.h5' with your actual model file)
model = load_model("model.h5")

# Define class labels (Adjust based on your dataset)
CLASS_NAMES = ["Normal", "Glaucoma", "Diabetic Retinopathy", "Cataract"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file temporarily
    file_path = os.path.join("static/uploads", file.filename)
    file.save(file_path)

    # Process image
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))  # Adjust size based on model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
