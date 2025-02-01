import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from tqdm import tqdm

# These are some machine learning methods (including supervised and unsupervised) and some deep learning methods proposed by kaggle and standford cheat sheet
df = pd.read_csv('train.csv')
# missing_value
missing_val = df.isnull().sum()  # no missing values
print(df.nunique())
# first part: feature engineering
df['Gender'] = df['Gender'].map({
    'Female':1,
    'Male':0
})
df['Vehicle_Damage'] = df['Vehicle_Damage'].map({
    'Yes':1,
    'No': 0
})
# if it has the meaningful order
# if not meaningful, red, blue, green. We should use the dummy variable
df['Vehicle_Age'] = df['Vehicle_Age'].map({
    '< 1 Year': 0,
    '1-2 Year': 1,
    '> 2 Years': 2
})

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
# Train and score baseline model
for i in tqdm(range(100),desc='Processing'):  
    baseline = RandomForestRegressor(criterion="absolute_error", random_state=0)
    baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
    baseline_score = -1 * baseline_score.mean()

    print(f"MAE Baseline Score: {baseline_score:.4}")