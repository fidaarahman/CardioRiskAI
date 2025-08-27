import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="heart_disease_model_with_metadata.tflite")

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details
print("Input Details:")
for detail in input_details:
    print(detail)

# Print output details
print("\nOutput Details:")
for detail in output_details:
    print(detail)

# Check number of tensors
print("\nNumber of tensors in the model:", len(interpreter.get_tensor_details()))

# Check model signature
try:
    print("\nSignature:", interpreter.get_signature_list())
except Exception as e:
    print("No Signature Metadata:", e)
