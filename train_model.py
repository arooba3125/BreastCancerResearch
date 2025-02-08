import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Load dataset
data = pd.read_csv('E:/third semester/ProbabilityStatistics/Probability project/smote_undersampled_data.csv')

# Encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Define features and target
X = data.drop(['diagnosis_encoded'], axis=1)
y = data['diagnosis_encoded']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Hyperparameter tuning grid
param_grid = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Train the best model
rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=20, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

# Save the trained model
best_model = random_search.best_estimator_
joblib.dump(best_model, 'best_rf_model.pkl')

print("✅ Model training completed and saved!")
