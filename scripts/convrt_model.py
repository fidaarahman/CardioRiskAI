import tensorflow as tf

# Path to your model
model_path = 'E:/downloads/Cardiovascular-Disease-classification-master/hlo.h5'

# Load the model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except ValueError as e:
    print(f"Error loading model: {e}")
    model = None  # Make sure 'model' is set to None if loading fails

# Only proceed with conversion if the model is loaded successfully
if model is not None:
    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optionally, apply optimizations (e.g., quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert and save the model as a .tflite file
    tflite_model = converter.convert()

    # Path to save the .tflite model
    tflite_model_path = 'E:/downloads/Cardiovascular-Disease-classification-master/model.tflite'

    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model successfully converted to TensorFlow Lite: {tflite_model_path}")
else:
    print("Model loading failed, skipping TensorFlow Lite conversion.")
