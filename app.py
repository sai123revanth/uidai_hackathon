import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="UIDAI Analytics Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STEALTH CSS INJECTION ---
# Custom CSS to manage layout, hide Streamlit branding, and ensure scrollability.
st.markdown("""
    <style>
    /* Ensure the main app allows scrolling and removes standard padding */
    .main {
        overflow: auto;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Hide the Streamlit Header and Footer */
    header { visibility: hidden; height: 0px !important; }
    footer { visibility: hidden !important; height: 0px !important; }
    #MainMenu { visibility: hidden; }

    /* Dark theme portal colors */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        color: #38bdf8 !important;
        font-weight: 700;
    }

    /* Scrollbar Styling for a premium look */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #38bdf8;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Cleaning Utilities ---
def clean_data(df):
    """Applies standardization and noise removal."""
    # 1. Remove "1000" related noise from State/District names
    df = df[~df['state'].astype(str).str.contains('1000', na=False)]
    if 'district' in df.columns:
        df = df[~df['district'].astype(str).str.contains('1000', na=False)]

    # 2. Standardize State Names
    def standardize(name):
        if not isinstance(name, str): return name
        name = name.strip().title().replace("&", "And")
        mapping = {
            "Andaman And Nicobar Islands": "Andaman & Nicobar Islands",
            "A & N Islands": "Andaman & Nicobar Islands",
            "Dadra And Nagar Haveli And Daman And Diu": "D&N Haveli and Daman & Diu",
            "Jammu And Kashmir": "Jammu & Kashmir"
        }
        for key, val in mapping.items():
            if name.lower() == key.lower(): return val
        return name.replace(" And ", " & ")

    df['state'] = df['state'].apply(standardize)
    return df

# --- Data Loading ---
@st.cache_data(show_spinner=False)
def load_data(file_path):
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path, low_memory=False)
    
    # Clean noise and standardize names
    df = clean_data(df)
    
    # Parse dates and numbers
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce', dayfirst=True)
    cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['total'] = df[cols].sum(axis=1)
    return df.dropna(subset=['date'])

# --- Application Logic ---
FILE_NAME = "api_data_aadhar_enrolment_0_500000.csv"
df = load_data(FILE_NAME)

if df is not None:
    # Header Section
    st.title("🇮🇳 UIDAI Enrollment Analytics")
    st.caption("Strategic insight into demographic registration patterns across India.")
    
    # Filters
    selected_state = st.sidebar.selectbox("State Filter", ["All States"] + sorted(df['state'].unique().tolist()))
    f_df = df if selected_state == "All States" else df[df['state'] == selected_state]
    
    # Top-level Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Total Enrolled", f"{f_df['total'].sum():,.0f}")
    with m2: st.metric("Child (0-5)", f"{f_df['age_0_5'].sum():,.0f}")
    with m3: st.metric("Youth (5-17)", f"{f_df['age_5_17'].sum():,.0f}")
    with m4: st.metric("Adults (18+)", f"{f_df['age_18_greater'].sum():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualization Row
    l, r = st.columns([2, 1])
    with l:
        trend = f_df.groupby('date')['total'].sum().reset_index()
        fig_trend = px.area(trend, x='date', y='total', template="plotly_dark", 
                            color_discrete_sequence=['#38bdf8'], title="Registration Velocity")
        fig_trend.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with r:
        pie_data = f_df[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
        fig_pie = px.pie(values=pie_data.values, names=['Infant', 'Youth', 'Adult'], 
                         hole=0.6, template="plotly_dark", title="Demographic Split")
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Scrollable Data Explorer ---
    st.markdown("### 📋 Raw Data Explorer")
    st.info("The table below is interactive and scrollable (both vertically and horizontally).")
    # Streamlit's st.dataframe automatically includes scroll bars for large datasets.
    st.dataframe(
        f_df[['date', 'state', 'age_0_5', 'age_5_17', 'age_18_greater', 'total']], 
        use_container_width=True, 
        hide_index=True
    )

else:
    st.error("Data source missing. Please ensure 'api_data_aadhar_enrolment_0_500000.csv' is in the root directory.")