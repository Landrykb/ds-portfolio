from pyodide.http import open_url
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(open_url("/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)
cat_cols = df.select_dtypes(include='object').columns.drop(['customerID', 'Churn'])
df_encoded = pd.get_dummies(df.drop(['customerID', 'Churn'], axis=1), columns=cat_cols, drop_first=True)
X = df_encoded.drop('Churn_binary', axis=1)
y = df_encoded['Churn_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Features: {X.shape[1]}")
print(f"Train churn rate: {y_train.mean()*100:.1f}%")
