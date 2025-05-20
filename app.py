# app.py (Flask Backend)
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('asl_model.h5')  # Ensure this matches your model

# Check model input shape
input_shape = model.input_shape[1:]

# Assuming the labels are A-Z (adjust if different)
LABELS = [chr(i) for i in range(65, 91)]  # 'A' to 'Z'

# Flask setup
app = Flask(__name__)
CORS(app)

# Adjust this function based on model input shape
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((input_shape[0], input_shape[1]))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape((1, *input_shape))  # (1, height, width, channels)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_bytes = request.files['image'].read()
        img_array = preprocess_image(image_bytes)
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        confidence = float(preds[0][pred_index])
        pred_label = LABELS[pred_index] if pred_index < len(LABELS) else str(pred_index)

        return jsonify({
            'prediction': pred_label,
            'confidence': confidence
        })
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)