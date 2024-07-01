import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('data.csv')

# Delete Columns Deletion on lead_time
df = df.drop(['lead_time'], axis=1)

# Duplicate values
df = df.drop_duplicates()

# Feature Deletion
df = df.drop(['potential_issue', 'perf_6_month_avg', 'lead_time', 'sku'], axis=1)

# One-hot encoding
encoder = OneHotEncoder(sparse=False)
df[['potential_issue', 'deck_risk']] = encoder.fit_transform(df[['potential_issue', 'deck_risk']])

# Balancing data using Over Sampling
le = LabelEncoder()
y = le.fit_transform(df['went_on_backorder'])
X = df.drop('went_on_backorder', axis=1)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=2, random_state=1, shuffle=True, weights=[0.9, 0.1])
smote = SMOTE()
X, y = smote.fit_resample(X, y)
df = pd.concat([X, pd.DataFrame(y)], axis=1)
df.columns = df.columns.astype(str)

# Normalization using Z-Score method
scaler = StandardScaler()
df['in_transit_qty'] = scaler.fit_transform(df[['in_transit_qty']])

# Outliers using Z-Score method
z_scores = np.abs(stats.zscore(df['national_inv']))
df = df[z_scores < 3]

# Save the transformed data frame as cleaned_data.csv
df.to_csv('cleaned_data.csv', index=False)

# Print the top 5 rows of cleaned dataframe
print(df.head())