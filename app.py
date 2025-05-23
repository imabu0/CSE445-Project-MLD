import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model and class names
model = load_model("asl_model.h5")
class_names = np.load("class_names.npy", allow_pickle=True)

# Parameters
IMG_SIZE = (64, 64)  # Must match training size

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a window with a larger size
cv2.namedWindow("ASL Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ASL Recognition", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Get hand region (you can modify this to use hand detection)
    # For simplicity, we'll use a fixed region
    height, width = frame.shape[:2]
    x1, y1 = int(width*0.25), int(height*0.25)
    x2, y2 = int(width*0.75), int(height*0.75)
    
    # Draw rectangle for hand placement guidance
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Extract hand region
    hand_region = frame[y1:y2, x1:x2]
    
    if hand_region.size != 0:
        # Preprocess the hand region
        resized = cv2.resize(hand_region, IMG_SIZE)
        normalized = resized.astype('float32') / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        
        # Make prediction
        predictions = model.predict(input_tensor)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Display prediction
        label = f"{class_names[predicted_class]}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display instructions
    cv2.putText(frame, "Place your hand in the green box", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'Q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("ASL Recognition", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()