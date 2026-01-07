from flask import Flask, jsonify, render_template
from flask_cors import CORS  # Added for cross-origin support
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

app = Flask(__name__)
CORS(app) # This allows your frontend to talk to the API from any domain

# --- CONFIGURATION ---
CSV_FILE = 'api_data_aadhar_enrolment_500000_1000000.csv'

def load_and_clean_data():
    if not os.path.exists(CSV_FILE):
        return None
    
    df = pd.read_csv(CSV_FILE, low_memory=False)
    df.columns = df.columns.str.strip()
    
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['date'])
    
    numeric_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    df['month_year'] = df['date'].dt.to_period('M').astype(str)
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/visual/treemap')
def visual_treemap():
    df = load_and_clean_data()
    if df is None: return "CSV file missing", 404
    
    df_treemap = df[df['age_0_5'] > 0].copy()
    fig = px.treemap(
        df_treemap,
        path=[px.Constant("India"), 'state', 'district'],
        values='age_0_5',
        color='age_0_5',
        color_continuous_scale='RdYlGn',
        title='Early-Child Enrolment Density (Identifying Low-Enrolment Pockets)',
        template="plotly_white"
    )
    fig.update_layout(margin=dict(t=30, l=10, r=10, b=10))
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

@app.route('/visual/trend')
def visual_trend():
    df = load_and_clean_data()
    if df is None: return "CSV file missing", 404
    
    summary = df.groupby('district')['age_0_5'].sum().sort_values()
    focus_districts = list(summary.head(3).index) + list(summary.tail(3).index)
    
    trend_data = df[df['district'].isin(focus_districts)]
    trend_agg = trend_data.groupby(['month_year', 'district'])['age_0_5'].sum().reset_index()
    
    fig = px.line(
        trend_agg,
        x='month_year',
        y='age_0_5',
        color='district',
        markers=True,
        title='Growth Slope: Lagging vs. Leading Districts',
        template="plotly_white"
    )
    fig.update_layout(margin=dict(t=30, l=10, r=10, b=10))
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

@app.route('/api/summary')
def api_summary():
    df = load_and_clean_data()
    if df is None: return jsonify({"error": "Data not found"}), 404
    summary = df.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().to_dict(orient='index')
    return jsonify(summary)

@app.route('/api/inequality')
def api_inequality():
    df = load_and_clean_data()
    if df is None: return jsonify({"error": "Data not found"}), 404
    state_variance = df.groupby('state')['age_0_5'].std().reset_index()
    state_variance.columns = ['state', 'variance_std']
    top_5 = state_variance.sort_values(by='variance_std', ascending=False).head(5)
    return jsonify(top_5.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)