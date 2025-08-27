import tensorflow as tf

# Step 1: Load the existing model (.h5 file)
model = tf.keras.models.load_model('model.h5')

# Step 2: Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Step 3: Save the converted model as model.tflite
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite!")
