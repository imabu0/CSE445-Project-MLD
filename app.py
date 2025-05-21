import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector

# Constants
IMG_SIZE = 64  # Match your model's training size
OFFSET = 20

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Confirm model input shape
expected_shape = input_details[0]['shape']
print("Model expects input shape:", expected_shape)

# Load label names
with open("labels.txt", "r") as f:
    class_names = [line.strip().split(":")[1] for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access webcam.")
    exit()

# Hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from webcam.")
        continue

    # Flip the webcam image (to fix reversed view)
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop with padding
        x1 = max(0, x - OFFSET)
        y1 = max(0, y - OFFSET)
        x2 = min(img.shape[1], x + w + OFFSET)
        y2 = min(img.shape[0], y + h + OFFSET)
        imCrop = img[y1:y2, x1:x2]

        try:
            imResized = cv2.resize(imCrop, (IMG_SIZE, IMG_SIZE))
            imResized = imResized.astype(np.float32) / 255.0
            imResized = np.expand_dims(imResized, axis=0)

            # Set input and run prediction
            interpreter.set_tensor(input_details[0]['index'], imResized)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            class_index = np.argmax(prediction)
            predicted_label = class_names[class_index]

            # Display prediction on screen
            cv2.putText(img, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Show cropped hand
            cv2.imshow("Cropped Hand", imCrop)

        except Exception as e:
            print("Prediction error:", e)

    cv2.imshow("ASL Sign Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
