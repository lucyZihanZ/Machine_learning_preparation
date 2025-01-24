import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# These are some machine learning methods (including supervised and unsupervised) and some deep learning methods proposed by kaggle and standford cheat sheet

# data loading
df = pd.read_excel('Concrete_Data.xls')
# missing_value
missing_val = df.isnull().sum()  # no missing values
#  
print(df)

# first part: feature engineering
# First, we will establish a baseline by training the model on the un-augmented dataset. This will help us determine whether our new features are actually useful.
# We need to distinguish: classfication & prediction models
X = df.copy()
y = X.pop("Concrete compressive strength(MPa, megapascals)")

# Train and score baseline model
baseline = RandomForestRegressor(criterion="absolute_error", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()

print(f"MAE Baseline Score: {baseline_score:.4}")