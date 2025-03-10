import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from flask import Flask, request, jsonify

# Load dataset
file_path = r"C:\Users\MANUDEESH\Downloads\creditcard.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Separate features & target
X = df.drop(columns=['Class'])
y = df['Class']

# Split data before oversampling (avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Isolation Forest Model
iso_forest = IsolationForest(contamination=0.001, random_state=42)
iso_forest.fit(X_train_resampled)
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]

# Autoencoder Model
input_dim = X_train.shape[1]
autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(input_dim, activation="linear")  # Reconstruct input
])

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_train_resampled, X_train_resampled, epochs=10, batch_size=64, shuffle=True, validation_split=0.2)

# Get reconstruction error
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)  # Top 5% anomalies
y_pred_auto = [1 if x > threshold else 0 for x in mse]

# Evaluation
print("Isolation Forest:", classification_report(y_test, y_pred_iso))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_iso))

print("Autoencoder:", classification_report(y_test, y_pred_auto))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_auto))

# Flask API for fraud detection
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Fraud Detection API! Use the /predict endpoint for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  # Example: {'features': [0.1, 0.5, ...]}
    prediction = iso_forest.predict([data])[0]
    return jsonify({'fraud_probability': int(prediction == -1)})

if __name__ == '__main__':
    app.run(debug=True)