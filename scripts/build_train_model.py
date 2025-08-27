import pandas as pd
import numpy as np
import tensorflow as tf
import json
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tflite_support.metadata import MetadataWriter
from tflite_support.metadata import metadata_schema_pb2 as schema_fb

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=column_names)

# Replace '?' with NaN and convert to numeric
data.replace('?', pd.NA, inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')

# Fill missing values with column mean
data = data.fillna(data.mean())

# Split features and target
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# 2. Build and Train the Model
# ============================
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

model.save('heart_disease_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted TFLite model
tflite_model_path = 'heart_disease_model_with_metadata.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# =================================
# 4. Embed Dataset into Model File
# =================================
# Convert dataset to JSON
dataset = {
    'X_train': X_train.tolist(),
    'X_test': X_test.tolist(),
    'y_train': y_train.tolist(),
    'y_test': y_test.tolist()
}
dataset_json = json.dumps(dataset)

# Save dataset inside an HDF5 file
with h5py.File('heart_disease_model_with_data.h5', 'w') as hf:
    hf.create_dataset('dataset', data=dataset_json)

# ==================================
# 5. Add Metadata for Input/Output
# ==================================
# Create metadata object
metadata = schema_fb.ModelMetadata()
metadata.name = "Heart Disease Prediction Model"
metadata.description = "A model for predicting heart disease risk."
metadata.version = "1.0"

# Prepare input/output metadata
input_metadata = schema_fb.TensorMetadata()
input_metadata.name = "Input Features"
input_metadata.description = "Input features used for prediction"
input_metadata.tensor = schema_fb.TensorType.FLOAT32
input_metadata.shape.extend([1, X_train.shape[1]])

output_metadata = schema_fb.TensorMetadata()
output_metadata.name = "Prediction Result"
output_metadata.description = "Prediction result indicating heart disease risk"
output_metadata.tensor = schema_fb.TensorType.FLOAT32
output_metadata.shape.extend([1, 1])

# Create the metadata writer
writer = MetadataWriter.create_for_inference(
    model_content=tflite_model,
    input_norm_mean=[0.0],
    input_norm_std=[1.0],
    input_name="Input Features",
    output_name="Prediction Result"
)

# Save the model with embedded metadata
tflite_model_with_metadata = writer.write_metadata_to_model(tflite_model)

# Save the new model with metadata
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model_with_metadata)

print("Model with metadata saved successfully!")
