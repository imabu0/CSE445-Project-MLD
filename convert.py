import tensorflow as tf

# Load your trained Keras model
model = tf.keras.models.load_model("asl_model.h5")

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional
tflite_model = converter.convert()

# Save the converted model
with open("model_unquant.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted and saved as model_unquant.tflite")
