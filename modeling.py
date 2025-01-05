import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('E:/third semester/ProbabilityStatistics/undersampled_data.csv')

# Encode categorical data
categorical_cols = data.select_dtypes(include=['object']).columns  # finds all columns that are 'object' type
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Define the features and target
X = data.drop(['diagnosis_encoded'], axis=1)
y = data['diagnosis_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train)

# Streamlit app
st.title('Cancer Type Prediction Tool')
st.write('This tool predicts whether the cancer is benign or malignant based on input features.')

# User inputs for each feature
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f'Enter {feature}', value=float(X[feature].mean()))

# Button to make prediction
if st.button('Predict Cancer Type'):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)  # scale the input
    prediction = classifier.predict(input_scaled)
    prediction_text = 'Malignant' if prediction[0] == 1 else 'Benign'
    st.subheader(f'The predicted cancer type is: {prediction_text}')

# Visualize feature importances
features = list(X.columns)
importances = classifier.feature_importances_
indices = np.argsort(importances)
fig, ax = plt.subplots()
ax.barh(range(len(indices)), importances[indices], color='b')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([features[i] for i in indices])
ax.set_xlabel('Relative Importance')
ax.set_title('Feature Importances')
st.pyplot(fig)
