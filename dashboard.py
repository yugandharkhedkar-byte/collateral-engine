import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import requests
import json
import google.generativeai as genai

st.set_page_config(page_title="Collateral Valuation Engine", layout="wide", page_icon="🏠")

# ── Load models ──
@st.cache_resource
def load_models():
    with open("models/price_model.pkl", "rb") as f:
        price_model = pickle.load(f)
    with open("models/resale_model.pkl", "rb") as f:
        resale_model = pickle.load(f)
    with open("models/label_encoders.pkl", "rb") as f:
        le_dict = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return price_model, resale_model, le_dict, feature_cols

price_model, resale_model, le_dict, feature_cols = load_models()

def get_fraud_flags(property_type, size_sqft, area, city, floor_level, legal_status, rental_yield_pct, occupancy_status):
    flags = []

    # Size sanity checks per property type
    size_limits = {
        "Apartment":  (200,  5000),
        "Villa":      (800,  15000),
        "Plot":       (500,  20000),
        "Shop":       (50,   3000),
        "Warehouse":  (1000, 50000),
    }
    min_size, max_size = size_limits[property_type]
    if size_sqft < min_size:
        flags.append(f"🚨 Size too small for {property_type} — {size_sqft} sqft is below minimum {min_size} sqft")
    if size_sqft > max_size:
        flags.append(f"🚨 Size unusually large for {property_type} — {size_sqft} sqft exceeds typical {max_size} sqft")

    # Floor level mismatch
    if property_type in ["Villa", "Plot", "Warehouse"] and floor_level > 0:
        flags.append(f"🚨 Floor level {floor_level} is invalid for {property_type} — should be ground floor only")

    # Rental yield without rented status
    if rental_yield_pct > 0 and occupancy_status != "Rented":
        flags.append(f"🚨 Rental yield {rental_yield_pct}% entered but occupancy is '{occupancy_status}' — mismatch detected")

    # Disputed title with high rental yield
    if legal_status == "Disputed" and rental_yield_pct > 3:
        flags.append("🚨 High rental yield on disputed property — verify ownership before proceeding")

    # Mumbai plot size check
    if city == "Mumbai" and property_type == "Plot" and size_sqft > 5000:
        flags.append("🚨 Plot size unusually large for Mumbai — verify land records")

    # Shop on high floor
    if property_type == "Shop" and floor_level > 3:
        flags.append(f"🚨 Shop on floor {floor_level} — commercial viability and footfall risk")

    return flags

# ── Styling ──
st.markdown("""
<style>
.metric-card {
    background: #f8fafc;
    border-radius: 12px;
    padding: 20px;
    border-left: 5px solid #0d9488;
    margin-bottom: 12px;
    min-height: 100px;
}
.metric-value { font-size: 28px; font-weight: 700; color: #0f172a; }
.metric-label { font-size: 13px; color: #64748b; margin-bottom: 4px; }
.risk-card {
    background: #f8fafc;
    border-radius: 12px;
    padding: 20px;
    border-left: 5px solid #0d9488;
    color: #0f172a;
    min-height: 100px;
}
.narration-card {
    background: #f0fdf4;
    border-radius: 12px;
    padding: 20px;
    border-left: 5px solid #16a34a;
    font-size: 15px;
    line-height: 1.7;
    color: #14532d;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 AI-Powered Collateral Valuation Engine")
st.caption("Bloomberg Terminal for Real Estate Collateral — Price · Liquidity · Risk in one view")

st.divider()

# ── Input Form ──
st.subheader("📋 Property Details")
col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox("City", ["Pune", "Mumbai"])
    area_options = {
        "Pune": ["Baner", "Hinjewadi", "Kothrud", "Wakad", "Viman Nagar"],
        "Mumbai": ["Andheri West", "Powai", "Thane West", "Borivali", "Kharghar"]
    }
    area = st.selectbox("Area", area_options[city])
    property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Plot", "Shop", "Warehouse"])

with col2:
    size_sqft = st.number_input("Size (sq ft)", min_value=100, max_value=15000, value=850)
    age_bucket = st.selectbox("Property Age", ["New (<5yr)", "Mid (5-15yr)", "Old (>15yr)"])
    legal_status = st.selectbox("Legal Status", ["Clear", "Leasehold", "Disputed"])

with col3:
    occupancy_status = st.selectbox("Occupancy", ["Self-occupied", "Rented", "Vacant"])
    floor_level = st.slider("Floor Level", 0, 25, 3)
    rental_yield_pct = st.number_input("Rental Yield % (0 if not rented)", min_value=0.0, max_value=10.0, value=0.0)

has_lift = 1 if floor_level > 3 else 0

circle_map = {
    "Baner": 8500, "Hinjewadi": 7200, "Kothrud": 9800,
    "Wakad": 7800, "Viman Nagar": 10500, "Andheri West": 18000,
    "Powai": 16500, "Thane West": 12000, "Borivali": 13500, "Kharghar": 9000
}
infra_map = {
    "Baner": 82, "Hinjewadi": 75, "Kothrud": 88, "Wakad": 76,
    "Viman Nagar": 90, "Andheri West": 92, "Powai": 89,
    "Thane West": 83, "Borivali": 85, "Kharghar": 78
}
demand_map = {
    "Baner": 78, "Hinjewadi": 72, "Kothrud": 85, "Wakad": 74,
    "Viman Nagar": 88, "Andheri West": 90, "Powai": 87,
    "Thane West": 80, "Borivali": 82, "Kharghar": 75
}

analyze = st.button("🔍 Analyse Property", type="primary", use_container_width=True)

if analyze:
    # ── Build input row ──
    input_data = {
        "circle_rate_per_sqft": circle_map[area],
        "infra_score": infra_map[area],
        "demand_score": demand_map[area],
        "size_sqft": size_sqft,
        "floor_level": floor_level,
        "has_lift": has_lift,
        "rental_yield_pct": rental_yield_pct,
        "area_enc": le_dict["area"].transform([area])[0],
        "city_enc": le_dict["city"].transform([city])[0],
        "property_type_enc": le_dict["property_type"].transform([property_type])[0],
        "age_bucket_enc": le_dict["age_bucket"].transform([age_bucket])[0],
        "legal_status_enc": le_dict["legal_status"].transform([legal_status])[0],
        "occupancy_status_enc": le_dict["occupancy_status"].transform([occupancy_status])[0],
    }
    input_df = pd.DataFrame([input_data])[feature_cols]

    # ── Predictions ──
    market_value = float(price_model.predict(input_df)[0])
    resale_score = float(resale_model.predict(input_df)[0])
    resale_score = int(np.clip(resale_score, 0, 100))

    liquidity_discount = 1 - (0.35 - resale_score * 0.003)
    distress_value = market_value * liquidity_discount

    mv_low  = market_value * 0.93
    mv_high = market_value * 1.07
    dv_low  = distress_value * 0.93
    dv_high = distress_value * 1.07

    if resale_score >= 80:
        time_low, time_high = 20, 50
        liquidity_label = "🟢 Highly Liquid"
    elif resale_score >= 60:
        time_low, time_high = 45, 100
        liquidity_label = "🟡 Moderately Liquid"
    else:
        time_low, time_high = 90, 200
        liquidity_label = "🔴 Illiquid"

    confidence = round(np.clip(0.5 + (resale_score / 200) + (0.1 if legal_status == "Clear" else 0), 0.4, 0.95), 2)

    # ── Risk flags ──
    # ── Risk flags ──
    risk_flags = []
    if legal_status == "Disputed":
        risk_flags.append("⚠️ Disputed title — high legal risk")
    if legal_status == "Leasehold":
        risk_flags.append("⚠️ Leasehold property — limited resale appeal")
    if age_bucket == "Old (>15yr)":
        risk_flags.append("⚠️ Older construction — depreciation risk")
    if property_type in ["Warehouse", "Plot"]:
        risk_flags.append("⚠️ Niche asset — low buyer pool")
    if resale_score < 50:
        risk_flags.append("⚠️ Low resale score — exit risk")
    if not risk_flags:
        risk_flags.append("✅ No major risk flags detected")

    # ── Fraud checks ──
    fraud_flags = get_fraud_flags(
        property_type, size_sqft, area, city,
        floor_level, legal_status, rental_yield_pct, occupancy_status
    )
    st.divider()
    st.subheader("📊 Valuation Report")

    # ── Metric cards ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card" style="min-height:100px">
        <div class="metric-label">Market Value Range</div>
        <div class="metric-value">₹{mv_low/1e5:.1f}L – ₹{mv_high/1e5:.1f}L</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="metric-card" style="min-height:100px">
        <div class="metric-label">Distress Sale Value</div>
        <div class="metric-value">₹{dv_low/1e5:.1f}L – ₹{dv_high/1e5:.1f}L</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class="metric-card" style="min-height:100px">
        <div class="metric-label">Resale Score</div>
        <div class="metric-value">{resale_score} / 100 &nbsp; {liquidity_label}</div>
        </div>""", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(f"""<div class="metric-card" style="min-height:100px">
        <div class="metric-label">Time to Sell</div>
        <div class="metric-value">{time_low}–{time_high} days</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""<div class="metric-card" style="min-height:100px">
        <div class="metric-label">Confidence Score</div>
        <div class="metric-value">{confidence}</div>
        </div>""", unsafe_allow_html=True)

    with c6:
        st.markdown(f"""<div class="metric-card" style="min-height:100px">
        <div class="metric-label">Risk Flags</div>
        <div style="font-size:13px; margin-top:6px; color:#0f172a">{"<br>".join(risk_flags)}</div>
        </div>""", unsafe_allow_html=True)

    # ── Fraud flags display ──
    if fraud_flags:
        st.divider()
        st.subheader("🚨 Fraud & Sanity Check Alerts")
        for flag in fraud_flags:
            st.error(flag)
    else:
        st.divider()
        st.success("✅ Fraud & Sanity Checks Passed — All inputs are within expected ranges")

    # ── Map embed ──
    st.divider()
    st.subheader("📍 Property Location")
    
    area_coords = {
        "Baner":        (18.5590, 73.7868),
        "Hinjewadi":    (18.5904, 73.7381),
        "Kothrud":      (18.5074, 73.8077),
        "Wakad":        (18.5975, 73.7618),
        "Viman Nagar":  (18.5679, 73.9143),
        "Andheri West": (19.1360, 72.8262),
        "Powai":        (19.1176, 72.9060),
        "Thane West":   (19.2183, 72.9781),
        "Borivali":     (19.2307, 72.8567),
        "Kharghar":     (19.0474, 73.0659),
    }
    
    lat, lon = area_coords[area]
    zoom = 14
    
    map_html = f"""
    <div style="border-radius:12px; overflow:hidden; border: 1px solid #e2e8f0;">
    <iframe
        width="100%"
        height="300"
        frameborder="0"
        scrolling="no"
        marginheight="0"
        marginwidth="0"
        src="https://www.openstreetmap.org/export/embed.html?bbox={lon-0.02},{lat-0.02},{lon+0.02},{lat+0.02}&layer=mapnik&marker={lat},{lon}"
        style="border:0">
    </iframe>
    <div style="padding:8px 12px; background:#f8fafc; font-size:12px; color:#64748b;">
        📍 {area}, {city} — Lat: {lat}, Lon: {lon}
    </div>
    </div>
    """
    st.markdown(map_html, unsafe_allow_html=True)

    # ── SHAP chart ──
    st.divider()
    st.subheader("🔍 What drove this valuation?")
    explainer = shap.TreeExplainer(price_model)
    shap_values = explainer.shap_values(input_df)
    feature_names_clean = [f.replace("_enc","").replace("_"," ").title() for f in feature_cols]
    shap_df = pd.DataFrame({
        "Feature": feature_names_clean,
        "SHAP Value": shap_values[0]
    }).sort_values("SHAP Value", key=abs, ascending=True).tail(8)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#dc2626" if v < 0 else "#0d9488" for v in shap_df["SHAP Value"]]
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel("Impact on Market Value (₹)")
    ax.set_title("Feature Impact (SHAP)")
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # ── GenAI Narration ──
    st.divider()
    st.subheader("🤖 AI Loan Decision Summary")

    prompt = f"""You are a senior credit analyst at an NBFC in India. Based on the following collateral valuation data, write a concise 3-sentence loan decision summary for a banker. Be direct, professional, and mention the key risk factors and recommendation.

Property: {property_type} in {area}, {city}
Size: {size_sqft} sq ft | Age: {age_bucket} | Legal: {legal_status} | Occupancy: {occupancy_status}
Market Value: ₹{market_value/1e5:.1f}L | Distress Value: ₹{distress_value/1e5:.1f}L
Resale Score: {resale_score}/100 ({liquidity_label}) | Time to sell: {time_low}-{time_high} days
Confidence: {confidence} | Risk Flags: {", ".join(risk_flags)}

Write the summary now:"""

    with st.spinner("Generating AI analysis..."):
        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            narration = response.text
        except:
            narration = f"This {age_bucket.lower()} {property_type.lower()} in {area} shows a resale score of {resale_score}/100 with an estimated market value of ₹{market_value/1e5:.1f}L. {'Legal clarity is a concern and should be verified before disbursement.' if legal_status != 'Clear' else 'Title is clear with no major legal concerns.'} {'Recommend APPROVE' if resale_score >= 60 and legal_status != 'Disputed' else 'Recommend REVIEW'} with standard LTV norms applicable for this asset class."

    st.markdown(f'<div class="narration-card">{narration}</div>', unsafe_allow_html=True)

    # ── Loan Decision Banner ──
    st.divider()
    if resale_score >= 60 and legal_status != "Disputed":
        st.success(f"✅ RECOMMENDATION: APPROVE — Max LTV 65% | Property shows adequate liquidity and collateral cover")
    elif resale_score >= 40:
        st.warning(f"⚠️ RECOMMENDATION: CONDITIONAL APPROVE — Additional due diligence required")
    else:
        st.error(f"❌ RECOMMENDATION: DECLINE — Insufficient liquidity or high risk profile")