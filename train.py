import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Dataset parameters
DATASET_PATH = "asl_dataset"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 30

# Load and preprocess data
def load_data(dataset_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))
    
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, IMG_SIZE)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                labels.append(class_names.index(class_name))
    
    return np.array(images), np.array(labels), class_names

# Load the dataset
print("Loading dataset...")
X, y, class_names = load_data(DATASET_PATH)

# Normalize pixel values
X = X.astype('float32') / 255.0

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=len(class_names))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# Create model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

model = create_model((IMG_SIZE[0], IMG_SIZE[1], 3), len(class_names))
model.summary()

# Callbacks
checkpoint = ModelCheckpoint("asl_model.h5", 
                           monitor='val_accuracy',
                           save_best_only=True,
                           mode='max',
                           verbose=1)

early_stop = EarlyStopping(monitor='val_accuracy',
                          patience=5,
                          restore_best_weights=True,
                          verbose=1)

# Train the model
print("Training model...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop]
)

# Save class names
np.save("class_names.npy", class_names)

print("Training completed. Model saved as asl_model.h5")