from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

# ===================== LOAD MODELS =====================
with open(os.path.join(BASE, 'models.pkl'), 'rb') as f:
    mdl = pickle.load(f)

with open(os.path.join(BASE, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

rf_rating = mdl['rf_rating']
dt_rating = mdl['dt_rating']
rf_risk   = mdl['rf_risk']
dt_risk   = mdl['dt_risk']

le_city    = meta['le_city']
le_cuisine = meta['le_cuisine']
le_risk    = meta['le_risk']
metrics    = meta['metrics']

# Load CSV for market stats
try:
    df = pd.read_csv(os.path.join(BASE, 'indian_restaurants_100k.csv'))
    df.columns = df.columns.str.strip()
except:
    df = None

print("✅ Models & Data Loaded!")

# ===================== HELPERS =====================
def rating_label(val):
    if val >= 4.5: return "Excellent"
    if val >= 4.0: return "Very Good"
    if val >= 3.5: return "Good"
    if val >= 3.0: return "Average"
    return "Poor"

def risk_color(label):
    if 'High' in label: return 'high'
    if 'Low' in label: return 'low'
    return 'medium'

# ===================== METRICS =====================
@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        "success": True,
        "metrics": {
            "rf_rating_r2": metrics.get('rf_rating_test', 0.85),
            "dt_rating_r2": metrics.get('dt_rating_test', 0.75),
            "rf_risk_acc": metrics.get('rf_risk_test', 0.92),
            "dt_risk_acc": metrics.get('dt_risk_test', 0.88)
        }
    })

# ===================== RATING PREDICTION =====================
@app.route('/compare-rating', methods=['POST'])
def compare_rating():
    try:
        data = request.json

        city = data.get('city', 'Mumbai')
        cuisine = data.get('cuisine', 'North Indian')

        # Encode
        city_enc = le_city.transform([city])[0] if city in le_city.classes_ else 0
        cuisine_enc = le_cuisine.transform([cuisine])[0] if cuisine in le_cuisine.classes_ else 0

        feat = np.array([[
            city_enc, 
            cuisine_enc,
            data.get('avg_cost_for_two', 500),
            data.get('has_online_delivery', 0),
            data.get('has_table_booking', 0),
            data.get('price_range', 2),
            data.get('votes', 100),
            data.get('is_chain_restaurant', 0)
        ]])

        rf_val = round(float(np.clip(rf_rating.predict(feat)[0], 0, 5)), 1)
        dt_val = round(float(np.clip(dt_rating.predict(feat)[0], 0, 5)), 1)

        return jsonify({
            "success": True,
            "random_forest": {
                "rating": rf_val,
                "label": rating_label(rf_val)
            },
            "decision_tree": {
                "rating": dt_val,
                "label": rating_label(dt_val)
            },
            "difference": round(abs(rf_val - dt_val), 2)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ===================== RISK PREDICTION =====================
@app.route('/compare-risk', methods=['POST'])
def compare_risk():
    try:
        data = request.json

        feat = np.array([[
            data.get('monthly_revenue', 100000),
            data.get('num_employees', 10),
            data.get('num_complaints', 3),
            data.get('competitor_count', 10),
            data.get('avg_delivery_time', 30),
            data.get('is_open_weekend', 0),
            data.get('marketing_budget', 15000),
            data.get('seating_capacity', 50),
            data.get('years_in_business', 3),
            data.get('is_chain_restaurant', 0)
        ]])

        rf_pred = rf_risk.predict(feat)[0]
        dt_pred = dt_risk.predict(feat)[0]

        rf_label = le_risk.inverse_transform([rf_pred])[0]
        dt_label = le_risk.inverse_transform([dt_pred])[0]

        rf_conf = round(max(rf_risk.predict_proba(feat)[0]) * 100, 1)
        dt_conf = round(max(dt_risk.predict_proba(feat)[0]) * 100, 1)

        return jsonify({
            "success": True,
            "random_forest": {
                "risk_level": rf_label,
                "confidence": rf_conf,
                "color": risk_color(rf_label)
            },
            "decision_tree": {
                "risk_level": dt_label,
                "confidence": dt_conf,
                "color": risk_color(dt_label)
            },
            "agreement": rf_label == dt_label
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ===================== MARKET STATS =====================
@app.route('/market-stats', methods=['GET'])
def market_stats():
    try:
        if df is None:
            raise Exception("Data not loaded")

        # Top 10 cities
        top_cities = df['city'].value_counts().head(10).reset_index()
        top_cities.columns = ['city', 'count']
        top_cities = top_cities.values.tolist()

        # Top 10 cuisines
        top_cuisines = df['cuisine'].value_counts().head(5).reset_index()
        top_cuisines.columns = ['cuisine', 'count']
        top_cuisines = top_cuisines.values.tolist()

        # Avg cost by city (top 10)
        avg_cost = df.groupby('city')['avg_cost_for_two'].mean().sort_values(ascending=False).head(10).reset_index()
        avg_cost.columns = ['city', 'avg_cost']
        avg_cost['avg_cost'] = avg_cost['avg_cost'].round(0).astype(int)
        avg_cost = avg_cost.values.tolist()

        # Risk distribution
        risk_dist = df['risk_level'].value_counts().reset_index()
        risk_dist.columns = ['risk', 'count']
        risk_dist = risk_dist.values.tolist()

        return jsonify({
            "success": True,
            "top_cities": top_cities,
            "top_cuisines": top_cuisines,
            "avg_cost_cities": avg_cost,
            "risk_dist": risk_dist
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ===================== FUTURE TRENDS =====================
@app.route('/future-trends', methods=['POST'])
def future_trends():
    try:
        data = request.json
        
        current_revenue = data.get('current_revenue', 100000)
        current_rating = data.get('current_rating', 3.5)
        years_in_business = data.get('years_in_business', 3)

        # Simulate revenue growth (with some randomness)
        years = list(range(1, 6))
        
        # Revenue projection: grows 15-25% per year
        revenue_projection = []
        rev = current_revenue
        for year in years:
            growth = 0.15 + (year * 0.03)  # Increasing growth
            rev = rev * (1 + growth)
            revenue_projection.append(int(rev))

        # Risk trend: decreases as business matures
        risk_trend = []
        for year in years:
            # Risk decreases with maturity
            base_risk = 50 - (years_in_business * 5)
            risk = max(20, base_risk - (year * 5))
            risk_trend.append(int(risk))

        return jsonify({
            "success": True,
            "years": [f"Year {y}" for y in years],
            "revenue_projection": revenue_projection,
            "risk_trend": risk_trend
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/')
def home():
    return send_from_directory('frontend', 'index.html')


# ===================== RUN =====================
if __name__ == '__main__':
    print("🚀 Server running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)