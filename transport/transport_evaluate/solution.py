from pyodide.http import open_url
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load & prepare data (steps 1-3)
df = pd.read_csv(open_url("/datasets/public_transport_delays.csv"))
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
# Combine date + time columns into a single timestamp
df['datetime']    = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
df['hour']        = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month']       = df['datetime'].dt.month
df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather_condition'])
df['event_encoded'] = le.fit_transform(df['event_type'].fillna('none'))
df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
df['weather_rush'] = df['weather_encoded'] * df['is_rush_hour']
df['event_peak'] = (df['event_encoded'] > 0).astype(int) * df['is_rush_hour']
feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                'is_rush_hour', 'weather_encoded', 'event_encoded',
                'weather_rush', 'event_peak', 'temperature_C']
X = df[feature_cols]
y = df['actual_departure_delay_min']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Train model & evaluate
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
print(f"RMSE: {rmse:.2f} minutes")
print(f"5-Fold CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
