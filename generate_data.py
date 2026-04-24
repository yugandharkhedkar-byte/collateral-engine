import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 200

locations = [
    {"area": "Baner",        "city": "Pune",   "circle_rate": 8500,  "infra_score": 82, "demand_score": 78},
    {"area": "Hinjewadi",    "city": "Pune",   "circle_rate": 7200,  "infra_score": 75, "demand_score": 72},
    {"area": "Kothrud",      "city": "Pune",   "circle_rate": 9800,  "infra_score": 88, "demand_score": 85},
    {"area": "Wakad",        "city": "Pune",   "circle_rate": 7800,  "infra_score": 76, "demand_score": 74},
    {"area": "Viman Nagar",  "city": "Pune",   "circle_rate": 10500, "infra_score": 90, "demand_score": 88},
    {"area": "Andheri West", "city": "Mumbai", "circle_rate": 18000, "infra_score": 92, "demand_score": 90},
    {"area": "Powai",        "city": "Mumbai", "circle_rate": 16500, "infra_score": 89, "demand_score": 87},
    {"area": "Thane West",   "city": "Mumbai", "circle_rate": 12000, "infra_score": 83, "demand_score": 80},
    {"area": "Borivali",     "city": "Mumbai", "circle_rate": 13500, "infra_score": 85, "demand_score": 82},
    {"area": "Kharghar",     "city": "Mumbai", "circle_rate": 9000,  "infra_score": 78, "demand_score": 75},
]

prop_types = ["Apartment", "Villa", "Plot", "Shop", "Warehouse"]
prop_weights = [0.55, 0.15, 0.15, 0.10, 0.05]

age_buckets = ["New (<5yr)", "Mid (5-15yr)", "Old (>15yr)"]
age_weights = [0.30, 0.45, 0.25]

legal_status = ["Clear", "Leasehold", "Disputed"]
legal_weights = [0.70, 0.20, 0.10]

occupancy = ["Self-occupied", "Rented", "Vacant"]
occ_weights = [0.45, 0.35, 0.20]

rows = []
for i in range(n):
    loc = locations[np.random.randint(0, len(locations))]
    ptype = np.random.choice(prop_types, p=prop_weights)
    age = np.random.choice(age_buckets, p=age_weights)
    legal = np.random.choice(legal_status, p=legal_weights)
    occ = np.random.choice(occupancy, p=occ_weights)

    if ptype == "Apartment":
        size = np.random.randint(450, 2000)
    elif ptype == "Villa":
        size = np.random.randint(1500, 5000)
    elif ptype == "Plot":
        size = np.random.randint(800, 4000)
    elif ptype == "Shop":
        size = np.random.randint(150, 800)
    else:
        size = np.random.randint(2000, 10000)

    floor = np.random.randint(0, 20) if ptype == "Apartment" else 0
    has_lift = 1 if floor > 3 else 0

    age_factor = {"New (<5yr)": 1.0, "Mid (5-15yr)": 0.88, "Old (>15yr)": 0.74}[age]
    legal_factor = {"Clear": 1.0, "Leasehold": 0.92, "Disputed": 0.80}[legal]
    type_factor = {"Apartment": 1.0, "Villa": 1.15, "Plot": 0.90, "Shop": 1.05, "Warehouse": 0.80}[ptype]
    occ_factor = {"Self-occupied": 1.0, "Rented": 1.04, "Vacant": 0.96}[occ]

    base_price_per_sqft = loc["circle_rate"] * np.random.uniform(1.05, 1.35)
    market_value = (base_price_per_sqft * size * age_factor * legal_factor * type_factor * occ_factor)
    market_value = round(market_value + np.random.normal(0, market_value * 0.04), -3)

    liquidity_score_raw = (
        loc["demand_score"] * 0.35 +
        loc["infra_score"] * 0.25 +
        (100 if ptype == "Apartment" else 70 if ptype == "Villa" else 60 if ptype == "Shop" else 50 if ptype == "Plot" else 30) * 0.20 +
        (100 if legal == "Clear" else 60 if legal == "Leasehold" else 20) * 0.20
    )
    resale_score = int(np.clip(liquidity_score_raw + np.random.normal(0, 4), 20, 98))

    liquidity_discount = 1 - (0.35 - resale_score * 0.003)
    distress_value = round(market_value * liquidity_discount, -3)

    if resale_score >= 80:
        time_to_sell = np.random.randint(20, 50)
    elif resale_score >= 60:
        time_to_sell = np.random.randint(45, 100)
    else:
        time_to_sell = np.random.randint(90, 200)

    rental_yield = round(np.random.uniform(2.5, 5.5), 2) if occ == "Rented" else 0.0
    confidence = round(np.clip(0.5 + (resale_score / 200) + (0.1 if legal == "Clear" else 0), 0.4, 0.95), 2)

    rows.append({
        "property_id": f"PROP{1000+i}",
        "area": loc["area"],
        "city": loc["city"],
        "circle_rate_per_sqft": loc["circle_rate"],
        "infra_score": loc["infra_score"],
        "demand_score": loc["demand_score"],
        "property_type": ptype,
        "size_sqft": size,
        "age_bucket": age,
        "floor_level": floor,
        "has_lift": has_lift,
        "legal_status": legal,
        "occupancy_status": occ,
        "rental_yield_pct": rental_yield,
        "market_value": market_value,
        "distress_value": distress_value,
        "resale_score": resale_score,
        "time_to_sell_days": time_to_sell,
        "confidence_score": confidence,
    })

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/properties.csv", index=False)
print(f"Dataset generated: {len(df)} rows")
print(df.head(3).to_string())