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

# --- ULTIMATE STEALTH CSS INJECTION ---
# This uses more aggressive selectors to ensure all Streamlit-specific UI elements are hidden.
st.markdown("""
    <style>
    /* Remove the white/gray border and padding around the main content */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Hide the Streamlit Header (the top bar) */
    header { visibility: hidden !important; height: 0px !important; }
    
    /* Hide the "Made with Streamlit" Footer and Watermark */
    footer { visibility: hidden !important; height: 0px !important; }
    div[data-testid="stStatusWidget"] { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    
    /* Hide the 'Built with Streamlit' link specifically if footer hiding isn't enough */
    .viewerBadge_container__1QSob { display: none !important; }
    .stDeployButton { display: none !important; }

    /* Match the background to your portal's dark theme */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    /* Style the metrics cards to look modern */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #38bdf8 !important;
    }
    
    /* Ensure the app takes the full height of the iframe without extra spacing */
    .main .block-container {
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data(show_spinner=False)
def load_data(file_path):
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        cols = ['age_0_5', 'age_5_17', 'age_18_greater']
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['total'] = df[cols].sum(axis=1)
        return df
    except Exception:
        return None

# --- Application Logic ---
FILE_NAME = "api_data_aadhar_enrolment_0_500000.csv"
df = load_data(FILE_NAME)

if df is not None:
    # Sidebar Filters
    st.sidebar.title("🎛️ Control Panel")
    selected_state = st.sidebar.selectbox("Select State Focus", ["All States"] + sorted(df['state'].unique().tolist()))
    
    # Filter Logic
    f_df = df if selected_state == "All States" else df[df['state'] == selected_state]
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Enrolled", f"{f_df['total'].sum():,.0f}")
    c2.metric("Child (0-5)", f"{f_df['age_0_5'].sum():,.0f}")
    c3.metric("Youth (5-17)", f"{f_df['age_5_17'].sum():,.0f}")
    c4.metric("Adults (18+)", f"{f_df['age_18_greater'].sum():,.0f}")

    st.markdown("---")
    
    # Visualizations
    l, r = st.columns([2, 1])
    with l:
        trend = f_df.groupby('date')['total'].sum().reset_index()
        fig_trend = px.area(trend, x='date', y='total', template="plotly_dark", 
                            color_discrete_sequence=['#38bdf8'], title="Registration Trends")
        fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_trend, use_container_width=True)
    with r:
        pie_data = f_df[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
        fig_pie = px.pie(values=pie_data.values, names=['0-5', '5-17', '18+'], 
                         hole=0.6, template="plotly_dark", title="Age Split")
        fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.error("Data source missing.")