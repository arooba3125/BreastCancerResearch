import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Step 1: Load the data
file_path = 'E:/third semester/ProbabilityStatistics/breast-cancer1.csv'
data = pd.read_csv(file_path)

# Step 2: Check the initial class distribution (optional)
print("Initial class distribution:")
print(data['diagnosis_encoded'].value_counts())

# Step 3: Separate features and target
X = data.drop('diagnosis_encoded', axis=1)  # Drop the target column to separate features
y = data['diagnosis_encoded']  # The target column

# Step 4: Perform undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Step 5: Check the new class distribution
print("New class distribution:")
print(pd.Series(y_resampled).value_counts())

# Step 6: Combine the resampled data back into a DataFrame
resampled_data = pd.concat([X_resampled, pd.Series(y_resampled, name='diagnosis_encoded')], axis=1)

# Step 7: Save the resampled data to a new CSV file (optional)
resampled_data.to_csv('E:/third semester/ProbabilityStatistics/undersampled_data.csv', index=False)
print("Undersampled data saved to 'undersampled_data.csv'")
