import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import shap

df = pd.read_csv("data/properties.csv")

# Encode categorical columns
le_dict = {}
cat_cols = ["area", "city", "property_type", "age_bucket", "legal_status", "occupancy_status"]
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    le_dict[col] = le

feature_cols = [
    "circle_rate_per_sqft", "infra_score", "demand_score",
    "size_sqft", "floor_level", "has_lift", "rental_yield_pct",
    "area_enc", "city_enc", "property_type_enc",
    "age_bucket_enc", "legal_status_enc", "occupancy_status_enc"
]

X = df[feature_cols]

# ── Model 1: Market Value ──
y_price = df["market_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y_price, test_size=0.2, random_state=42)
price_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
price_model.fit(X_train, y_train)
preds = price_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Price model MAE: ₹{mae:,.0f}")

# ── Model 2: Resale Score ──
y_resale = df["resale_score"]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_resale, test_size=0.2, random_state=42)
resale_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
resale_model.fit(X_train2, y_train2)
resale_preds = resale_model.predict(X_test2)
resale_mae = mean_absolute_error(y_test2, resale_preds)
print(f"Resale model MAE: {resale_mae:.2f} points")

# ── Save everything ──
os.makedirs("models", exist_ok=True)
with open("models/price_model.pkl", "wb") as f:
    pickle.dump(price_model, f)
with open("models/resale_model.pkl", "wb") as f:
    pickle.dump(resale_model, f)
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(le_dict, f)
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("Models saved successfully!")
print(f"Features used: {feature_cols}")