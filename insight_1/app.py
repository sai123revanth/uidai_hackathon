from flask import Flask, jsonify, render_template
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

app = Flask(__name__)

# --- CONFIGURATION ---
CSV_FILE = 'api_data_aadhar_enrolment_500000_1000000.csv'

def load_and_clean_data():
    """
    Standardized data pipeline to ensure consistency across all API endpoints.
    """
    if not os.path.exists(CSV_FILE):
        return None
    
    # Load with low_memory=False for large Aadhaar datasets
    df = pd.read_csv(CSV_FILE, low_memory=False)
    
    # Standardize column names (strip hidden spaces)
    df.columns = df.columns.str.strip()
    
    # Date processing
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Numeric processing
    numeric_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Time-series grouping key
    df['month_year'] = df['date'].dt.to_period('M').astype(str)
    
    return df

# --- ROUTES ---

@app.route('/')
def home():
    """Serves the dashboard frontend from the templates folder."""
    return render_template('index.html')

@app.route('/visual/treemap')
def visual_treemap():
    """Generates the interactive Treemap for Geographic Penetration Gaps."""
    df = load_and_clean_data()
    if df is None: return "CSV file missing", 404
    
    # Filtering age_0_5 > 0 to prevent ZeroDivisionError in Plotly
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
    """Generates the Slope Analysis comparing top and bottom performing districts."""
    df = load_and_clean_data()
    if df is None: return "CSV file missing", 404
    
    # Identify Top and Bottom 3 Districts based on volume
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
    """Returns JSON summary of enrolment numbers by state."""
    df = load_and_clean_data()
    if df is None: return jsonify({"error": "Data not found"}), 404
    
    summary = df.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().to_dict(orient='index')
    return jsonify(summary)

@app.route('/api/inequality')
def api_inequality():
    """Returns JSON list of states with the highest internal district inequality."""
    df = load_and_clean_data()
    if df is None: return jsonify({"error": "Data not found"}), 404
    
    # Calculate Standard Deviation of enrolment across districts within each state
    state_variance = df.groupby('state')['age_0_5'].std().reset_index()
    state_variance.columns = ['state', 'variance_std']
    top_5 = state_variance.sort_values(by='variance_std', ascending=False).head(5)
    
    return jsonify(top_5.to_dict(orient='records'))

if __name__ == '__main__':
    # Using threaded=True to handle multiple iframe requests simultaneously
    app.run(debug=True, port=5000, threaded=True)