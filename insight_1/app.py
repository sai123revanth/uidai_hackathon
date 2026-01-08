import pandas as pd
import numpy as np
from flask import Flask, render_template
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os

# BASE_DIR is .../UIDAI_HACKATHON/insight_1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TEMPLATE_DIR is .../UIDAI_HACKATHON/
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
# DATA_PATH is .../UIDAI_HACKATHON/api_data_aadhar_enrolment_0_500000.csv
DATA_PATH = os.path.join(TEMPLATE_DIR, 'api_data_aadhar_enrolment_0_500000.csv')

# Initialize Flask pointing to the root for templates
app = Flask(__name__, template_folder=TEMPLATE_DIR)

def get_plot_url():
    """Helper to convert matplotlib plots to base64 strings."""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

def process_insights():
    """Core analysis logic."""
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return None, None, None

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    # 1. State Metrics
    state_metrics = df.groupby('state').agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum',
        'total_enrolments': 'sum'
    }).reset_index()
    
    state_metrics['saturation_index'] = state_metrics['age_18_greater'] / (state_metrics['age_0_5'] + state_metrics['age_5_17'] + 1)
    state_metrics['bap_score'] = (state_metrics['age_0_5'] / state_metrics['total_enrolments']) * 100
    
    # 2. Temporal Data
    df['month'] = df['date'].dt.month_name()
    monthly_trend = df.groupby('month')['total_enrolments'].sum().reindex([
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
    ]).dropna()

    # 3. Pincode Hotspots
    pincode_analysis = df.groupby(['state', 'pincode']).agg({
        'total_enrolments': 'sum'
    }).reset_index()
    top_pincodes = pincode_analysis.sort_values(by='total_enrolments', ascending=False).head(12)

    return state_metrics, monthly_trend, top_pincodes

@app.route('/')
def index():
    state_metrics, monthly_trend, top_pincodes = process_insights()
    
    if state_metrics is None:
        return f"Error: Data file not found at {DATA_PATH}. Please check your folder structure."

    champions = state_metrics.sort_values(by='bap_score', ascending=False).head(5).to_dict('records')
    
    # Visual 1: Velocity
    plt.figure(figsize=(10, 4), facecolor='#0f172a')
    ax1 = sns.lineplot(x=monthly_trend.index, y=monthly_trend.values, marker='o', color='#38bdf8', linewidth=3)
    ax1.set_facecolor('#0f172a')
    plt.title('National Enrolment Velocity', color='white', fontsize=14)
    plt.xticks(rotation=45, color='#94a3b8')
    plt.yticks(color='#94a3b8')
    plt.tight_layout()
    plot_velocity = get_plot_url()
    plt.close()

    # Visual 2: Saturation
    plt.figure(figsize=(10, 4), facecolor='#0f172a')
    top_vol = state_metrics.sort_values(by='total_enrolments', ascending=False).head(10)
    ax2 = sns.barplot(data=top_vol, x='state', y='saturation_index', palette='Blues_d')
    ax2.set_facecolor('#0f172a')
    plt.title('Saturation Index (High-Volume States)', color='white', fontsize=14)
    plt.xticks(rotation=45, color='#94a3b8')
    plt.yticks(color='#94a3b8')
    plt.tight_layout()
    plot_saturation = get_plot_url()
    plt.close()

    return render_template(
        'insight_1.html', 
        champions=champions, 
        plot_v=plot_velocity,
        plot_s=plot_saturation,
        pincodes=top_pincodes.to_dict('records')
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)