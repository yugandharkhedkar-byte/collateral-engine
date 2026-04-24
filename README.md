# 🏠 AI-Powered Collateral Valuation & Resale Liquidity Engine

🚀 **Live Demo:** https://collateral-engine-uaifppqyn6ezzubxvgwemz.streamlit.app/

> A Bloomberg Terminal for Real Estate Collateral — Price, Liquidity, and Risk in one unified output.

## Problem
Banks and NBFCs rely on manual, inconsistent property valuations for loan decisions. 
This leads to mispriced risk, slow credit decisions, and conservative lending.

## Solution
An AI-powered collateral intelligence engine that gives lenders:
- ₹ Market Value Range
- ₹ Distress Sale Value
- Resale Potential Score (0–100)
- Time to Liquidate (days)
- Confidence Score
- Risk Flags with explainability
- Plain-English loan decision via GenAI

## Tech Stack
- **ML Models** — XGBoost for price prediction + resale scoring
- **Explainability** — SHAP feature importance charts
- **GenAI** — Google Gemini for plain-English loan decision narration
- **Frontend** — Streamlit dashboard
- **Data** — Synthetic dataset of 200 Pune/Mumbai properties

## How to Run
```bash
git clone https://github.com/yugandharkhedkar-byte/collateral-engine.git
cd collateral-engine
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
python train.py
streamlit run dashboard.py
```

## Project Structure

collateral-engine/
├── data/properties.csv       # Synthetic dataset
├── models/                   # Trained ML models
├── generate_data.py          # Dataset generation
├── train.py                  # ML training script
├── dashboard.py              # Streamlit UI
└── README.md

## Evaluation Alignment
| Criteria | Coverage |
|---|---|
| Valuation logic | XGBoost price model with location + property features |
| Liquidity modeling | Resale score + time-to-sell prediction |
| Feature depth | 13 features across location, property, legal, market |
| Explainability | SHAP charts + GenAI narration |
| Deployability | Runs locally, API-ready architecture |
