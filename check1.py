import pandas as pd

# Load dataset
file_path = "E:/third semester/ProbabilityStatistics/smote_undersampled_data.csv"  # Your balanced dataset
data = pd.read_csv(file_path)

# Check class distribution
print("Class distribution after SMOTE + Undersampling:")
print(data["diagnosis_encoded"].value_counts())
