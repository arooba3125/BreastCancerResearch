import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

# Load the data
file_path = r"E:\third semester\ProbabilityStatistics\breast-cancer1.csv"
data = pd.read_csv(file_path)

# Initial class distribution
print("Initial class distribution:")
print(data['diagnosis_encoded'].value_counts())

# 🔹 Convert categorical columns to numeric
for col in data.select_dtypes(include=['object']).columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Separate features and target
X = data.drop('diagnosis_encoded', axis=1)
y = data['diagnosis_encoded']

# Apply SMOTE (Oversampling)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Apply Random Undersampling (Reduce Over-represented Class)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_smote, y_smote)

# Check final class distribution
print("New class distribution after SMOTE + Undersampling:")
print(pd.Series(y_resampled).value_counts())

# Save the new dataset
resampled_data = pd.concat([X_resampled, pd.Series(y_resampled, name='diagnosis_encoded')], axis=1)
resampled_data.to_csv(r"E:\third semester\ProbabilityStatistics\smote_undersampled_data.csv", index=False)

print("SMOTE + Undersampled data saved to 'smote_undersampled_data.csv'")
