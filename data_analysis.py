import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Load dataset
df = pd.read_csv('train.csv')

# Check for missing values
missing_val = df.isnull().sum()
if missing_val.sum() > 0:
    print("Warning: There are missing values in the dataset!")

# Display unique values per column

# Feature engineering: Mapping categorical variables
df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})
df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
df['Vehicle_Age'] = df['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2})

# Drop remaining non-numeric columns (if any)
df = df.select_dtypes(include=[np.number])

# Define features (X) and target (y)
target_column = 'Response'  # Adjust if the actual target column has a different name
X = df.drop(columns=[target_column])
y = df[target_column]

X_data = pd.get_dummies(df['Vehicle_Age'], prefix=True)

# 
df.drop_duplicates()
# Train and evaluate the baseline model
baseline = RandomForestRegressor(criterion="squared_error", random_state=0)
baseline_score = cross_val_score(baseline, X, y, cv=5, scoring="neg_mean_absolute_error")

# Convert negative MAE to positive
baseline_score = -baseline_score.mean()

print(f"MAE Baseline Score: {baseline_score:.4f}")
