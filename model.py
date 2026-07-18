import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, mean_squared_error
import pickle
import os

# ================= LOAD DATA =================
CSV_PATH = "indian_restaurants_100k.csv"

try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Data Loaded! Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: {CSV_PATH} not found!")
    exit()

# ================= DATA CLEANING =================
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Numeric columns
num_cols = [
    'avg_cost_for_two', 'votes', 'monthly_revenue',
    'num_employees', 'num_complaints', 'competitor_count',
    'avg_delivery_time', 'marketing_budget', 'seating_capacity',
    'years_in_business', 'aggregate_rating'
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# Categorical columns
cat_cols = ['city', 'cuisine', 'risk_level']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')

# Binary columns
bin_cols = ['has_online_delivery', 'has_table_booking', 'is_open_weekend', 'is_chain_restaurant']
for col in bin_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

print("✅ Data Cleaned!")

# ================= ENCODING =================
le_city = LabelEncoder()
le_cuisine = LabelEncoder()
le_risk = LabelEncoder()

df['city_enc'] = le_city.fit_transform(df['city'])
df['cuisine_enc'] = le_cuisine.fit_transform(df['cuisine'])

if 'risk_level' in df.columns:
    df['risk_enc'] = le_risk.fit_transform(df['risk_level'])

print("✅ Encoding Done!")

# ================= RATING MODEL (REGRESSION) =================
print("\n📊 ===== TRAINING RATING MODELS =====")

features_rating = [
    'city_enc', 'cuisine_enc', 'avg_cost_for_two',
    'has_online_delivery', 'has_table_booking',
    'price_range', 'votes', 'is_chain_restaurant'
]

# Filter features that exist
features_rating = [f for f in features_rating if f in df.columns]

X_rating = df[features_rating]
y_rating = df['aggregate_rating'] if 'aggregate_rating' in df.columns else df['votes']

# Remove NaN
mask = ~(X_rating.isna().any(axis=1) | y_rating.isna())
X_rating = X_rating[mask]
y_rating = y_rating[mask]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_rating, y_rating, test_size=0.2, random_state=42
)

# Random Forest Rating
rf_rating = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_rating.fit(X_train_r, y_train_r)

rf_rating_pred_train = rf_rating.predict(X_train_r)
rf_rating_pred_test = rf_rating.predict(X_test_r)
rf_r2_train = r2_score(y_train_r, rf_rating_pred_train)
rf_r2_test = r2_score(y_test_r, rf_rating_pred_test)
rf_rmse = np.sqrt(mean_squared_error(y_test_r, rf_rating_pred_test))
rf_mae = mean_absolute_error(y_test_r, rf_rating_pred_test)

print(f"Random Forest Rating:")
print(f"  Train R²: {rf_r2_train:.3f} | Test R²: {rf_r2_test:.3f}")
print(f"  RMSE: {rf_rmse:.3f} | MAE: {rf_mae:.3f}")

# Decision Tree Rating
dt_rating = DecisionTreeRegressor(random_state=42)
dt_rating.fit(X_train_r, y_train_r)

dt_rating_pred_train = dt_rating.predict(X_train_r)
dt_rating_pred_test = dt_rating.predict(X_test_r)
dt_r2_train = r2_score(y_train_r, dt_rating_pred_train)
dt_r2_test = r2_score(y_test_r, dt_rating_pred_test)
dt_rmse = np.sqrt(mean_squared_error(y_test_r, dt_rating_pred_test))
dt_mae = mean_absolute_error(y_test_r, dt_rating_pred_test)

print(f"\nDecision Tree Rating:")
print(f"  Train R²: {dt_r2_train:.3f} | Test R²: {dt_r2_test:.3f}")
print(f"  RMSE: {dt_rmse:.3f} | MAE: {dt_mae:.3f}")

# ================= RISK MODEL (CLASSIFICATION) =================
print("\n📊 ===== TRAINING RISK MODELS =====")

features_risk = [
    'monthly_revenue', 'num_employees', 'num_complaints',
    'competitor_count', 'avg_delivery_time', 'is_open_weekend',
    'marketing_budget', 'seating_capacity', 'years_in_business',
    'is_chain_restaurant'
]

# Filter features that exist
features_risk = [f for f in features_risk if f in df.columns]

X_risk = df[features_risk]
y_risk = df['risk_enc'] if 'risk_enc' in df.columns else df['votes'] > df['votes'].median()

# Remove NaN
mask = ~(X_risk.isna().any(axis=1) | y_risk.isna())
X_risk = X_risk[mask]
y_risk = y_risk[mask]

X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
    X_risk, y_risk, test_size=0.2, random_state=42
)

# Random Forest Risk
rf_risk = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_risk.fit(X_train_k, y_train_k)

rf_risk_acc_train = accuracy_score(y_train_k, rf_risk.predict(X_train_k))
rf_risk_acc_test = accuracy_score(y_test_k, rf_risk.predict(X_test_k))

print(f"Random Forest Risk:")
print(f"  Train Accuracy: {rf_risk_acc_train:.3f} | Test Accuracy: {rf_risk_acc_test:.3f}")

# Decision Tree Risk
dt_risk = DecisionTreeClassifier(random_state=42)
dt_risk.fit(X_train_k, y_train_k)

dt_risk_acc_train = accuracy_score(y_train_k, dt_risk.predict(X_train_k))
dt_risk_acc_test = accuracy_score(y_test_k, dt_risk.predict(X_test_k))

print(f"\nDecision Tree Risk:")
print(f"  Train Accuracy: {dt_risk_acc_train:.3f} | Test Accuracy: {dt_risk_acc_test:.3f}")

# ================= SAVE MODELS =================
models = {
    "rf_rating": rf_rating,
    "dt_rating": dt_rating,
    "rf_risk": rf_risk,
    "dt_risk": dt_risk
}

meta = {
    "le_city": le_city,
    "le_cuisine": le_cuisine,
    "le_risk": le_risk,
    "metrics": {
        "rf_rating_train": round(rf_r2_train, 3),
        "rf_rating_test": round(rf_r2_test, 3),
        "rf_rating_rmse": round(rf_rmse, 3),
        "dt_rating_train": round(dt_r2_train, 3),
        "dt_rating_test": round(dt_r2_test, 3),
        "dt_rating_rmse": round(dt_rmse, 3),

        "rf_risk_train": round(rf_risk_acc_train, 3),
        "rf_risk_test": round(rf_risk_acc_test, 3),
        "dt_risk_train": round(dt_risk_acc_train, 3),
        "dt_risk_test": round(dt_risk_acc_test, 3)
    }
}

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "models.pkl"), "wb") as f:
    pickle.dump(models, f)
    print("\n✅ models.pkl saved!")

with open(os.path.join(BASE, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
    print("✅ meta.pkl saved!")

print("\n" + "="*50)
print("🎉 ALL MODELS TRAINED & SAVED SUCCESSFULLY!")
print("="*50)
print("\nTo run the app:")
print("  1. python model.py  (run this first to train models)")
print("  2. python app.py    (start Flask server)")
print("  3. Open http://127.0.0.1:5000 in browser")