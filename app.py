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

# --- Custom Highly Modern CSS ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    /* Hide Streamlit Header/Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Modern Card Container */
    div.element-container {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Plotly Chart Backgrounds */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data(show_spinner=False)
def load_data(file_path):
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['total'] = df[cols].sum(axis=1)
    return df

# --- Main Logic ---
FILE_NAME = "api_data_aadhar_enrolment_0_500000.csv"
df = load_data(FILE_NAME)

if df is not None:
    # Sidebar Filters
    st.sidebar.title("🎛️ Control Panel")
    selected_state = st.sidebar.selectbox("Select State Focus", ["All States"] + sorted(df['state'].unique().tolist()))
    
    # Filter Logic
    f_df = df if selected_state == "All States" else df[df['state'] == selected_state]
    
    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Enrolled", f"{f_df['total'].sum():,.00f}")
    c2.metric("Child (0-5)", f"{f_df['age_0_5'].sum():,.00f}")
    c3.metric("Youth (5-17)", f"{f_df['age_5_17'].sum():,.00f}")
    c4.metric("Adults (18+)", f"{f_df['age_18_greater'].sum():,.00f}")

    st.markdown("---")
    
    # Charts
    l, r = st.columns([2, 1])
    
    with l:
        # Time Trend
        trend = f_df.groupby('date')['total'].sum().reset_index()
        fig_trend = px.area(trend, x='date', y='total', template="plotly_dark",
                            color_discrete_sequence=['#0ea5e9'], title="Registration Velocity")
        fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with r:
        # Distribution
        pie_data = f_df[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
        fig_pie = px.pie(values=pie_data.values, names=['0-5', '5-17', '18+'], 
                         hole=0.6, template="plotly_dark", title="Age Demographics")
        fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    # District Ranking
    st.subheader("📍 High-Activity Districts")
    dist = f_df.groupby('district')['total'].sum().nlargest(10).reset_index()
    fig_dist = px.bar(dist, x='total', y='district', orientation='h', template="plotly_dark",
                      color='total', color_continuous_scale='Blues')
    fig_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_dist, use_container_width=True)

else:
    st.error("Data source missing.")