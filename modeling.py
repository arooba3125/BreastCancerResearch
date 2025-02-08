import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ----------------------------------------
# ✅ LOAD MODEL & SCALER
# ----------------------------------------

# Load saved model & scaler
try:
    best_model = joblib.load('best_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("✅ Model and Scaler Loaded Successfully!")
except FileNotFoundError:
    st.error("❌ Model or Scaler file not found. Please train the model using `train_model.py` first.")
    st.stop()

# Load dataset
file_path = "E:/third semester/ProbabilityStatistics/Probability project/smote_undersampled_data.csv"

try:
    data = pd.read_csv(file_path)
    st.success("✅ Dataset Loaded Successfully!")
except FileNotFoundError:
    st.error("❌ Dataset file not found. Please check the file path.")
    st.stop()

# Define features and target
X = data.drop(['diagnosis_encoded'], axis=1)
y = data['diagnosis_encoded']

# Split dataset (same split as training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale test features
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------
# ✅ MODEL PERFORMANCE METRICS
# ----------------------------------------

# Get predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Display model accuracy and classification report
st.subheader("📊 Model Performance Metrics")
st.write("### Classification Report:")
st.text(classification_report(y_test, y_pred))

# Display confusion matrix
st.write("### Confusion Matrix:")
fig, ax = plt.subplots()
conf_matrix = confusion_matrix(y_test, y_pred)
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# ----------------------------------------
# ✅ STREAMLIT APP - PREDICTION TOOL
# ----------------------------------------

st.title("Cancer Type Prediction Tool")
st.write("This tool predicts whether the cancer is **benign** or **malignant** based on input features.")

# User input for each feature
input_data = {}
for feature in X.columns:
    default_value = float(X[feature].mean().item())
    input_data[feature] = st.number_input(f"Enter {feature}", value=default_value)

# ----------------------------------------
# ✅ MAKE PREDICTIONS
# ----------------------------------------

if st.button("Predict Cancer Type"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)  # Scale input data

    # Get probability of Malignant (1) and apply threshold
    probability = best_model.predict_proba(input_scaled)[:, 1]
    threshold = 0.6  # Adjust threshold for sensitivity control
    prediction = (probability >= threshold).astype(int)

    prediction_text = "Malignant" if prediction[0] == 1 else "Benign"
    st.subheader(f"The predicted cancer type is: {prediction_text}")

# ----------------------------------------
# ✅ FEATURE IMPORTANCE VISUALIZATION
# ----------------------------------------

features = list(X.columns)
importances = best_model.feature_importances_
indices = np.argsort(importances)

fig, ax = plt.subplots()
ax.barh(range(len(indices)), importances[indices], color="b")
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([features[i] for i in indices])
ax.set_xlabel("Relative Importance")
ax.set_title("Feature Importances")

st.pyplot(fig)
