import numpy as np
import cv2
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Load labels
with open("labels.txt", "r") as f:
    class_names = [line.strip().split(":")[1] for line in f.readlines()]

# Load sample image (crop of your hand showing "A")
img = cv2.imread("sample.png")  # Replace with a test image
img = cv2.resize(img, (64, 64))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])

predicted_index = np.argmax(prediction)
print("Prediction:", class_names[predicted_index])
