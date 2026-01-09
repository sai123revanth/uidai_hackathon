import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aadhar Enrolment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ADVANCED CSS STYLING ---
st.markdown("""
    <style>
    /* Gradient Background for the entire App */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff;
    }
    
    /* Glassmorphism Card Style */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00CC96;
    }
    
    /* Typography */
    .h1-title {
        text-align: center;
        font-weight: 900;
        font-size: 2.8em;
        background: -webkit-linear-gradient(left, #00CC96, #3366FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        text-shadow: 0px 0px 20px rgba(0, 204, 150, 0.3);
    }
    .subtitle {
        text-align: center;
        color: #b0bec5;
        font-size: 1.1em;
        margin-bottom: 40px;
        font-weight: 300;
        line-height: 1.6;
    }
    
    /* Detailed Explanation Box */
    .explanation-box {
        background: rgba(0, 0, 0, 0.3);
        border-left: 5px solid #3366FF;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        font-size: 1em;
        color: #e0e0e0;
        line-height: 1.6;
    }
    
    /* Stat Highlights */
    .big-stat {
        font-size: 3em;
        font-weight: 800;
        color: #00CC96;
        line-height: 1;
    }
    .stat-label {
        color: #a0a0a0;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 4px;
        color: #fff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00CC96 !important;
        color: #000 !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_process_data():
    files = [
        "api_data_aadhar_enrolment_0_500000.csv",
        "api_data_aadhar_enrolment_500000_1000000.csv",
        "api_data_aadhar_enrolment_1000000_1006029.csv"
    ]
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except FileNotFoundError:
            return None

    df = pd.concat(dfs, ignore_index=True)
    
    # 1. Robust Date Parsing
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    # 2. Key Metrics
    df['total_enrolment'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    # 3. Standardization
    state_mapping = {
        'Westbengal': 'West Bengal', 'West  Bengal': 'West Bengal', 
        'West Bangal': 'West Bengal', 'Andhra Pradesh': 'Andhra Pradesh',
        'andhra pradesh': 'Andhra Pradesh', 'Odisha': 'Odisha',
        'ODISHA': 'Odisha', 'Orissa': 'Odisha', 'Pondicherry': 'Puducherry'
    }
    df['state'] = df['state'].str.strip().str.title().replace(state_mapping)
    df = df[df['state'] != '100000']
    
    # 4. Feature Engineering
    # Era Definition: The Core Insight
    df['Era'] = df['date'].apply(lambda x: 'Real-Time Era (Sept+)' if (x.year == 2025 and x.month >= 9) else 'Batch Era (Pre-Aug)')
    df['DayOfWeek'] = df['date'].dt.day_name()
    
    return df

# --- LOAD DATA ---
df = load_and_process_data()

if df is not None:
    
    # --- HEADER SECTION ---
    st.markdown("<h1 class='h1-title'>National Aadhar Enrolment Analytics</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='subtitle'>
        An interactive dashboard designed to uncover operational trends, visualize demographic distribution, 
        and analyze the transition from batch-based processing to real-time data reporting.
    </div>
    """, unsafe_allow_html=True)

    # --- CONTROL PANEL ---
    with st.container():
        st.markdown("### 🎛️ Data Filters")
        st.markdown("""
        Use these filters to narrow down the dataset. **'Reporting Era'** is particularly important as it separates the data 
        into two distinct phases of the project: the early 'Batch' phase and the later 'Real-Time' phase.
        """)
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_era = st.multiselect("Select Reporting Era:", df['Era'].unique(), default=df['Era'].unique())
        with col2:
            states = sorted(df['state'].unique())
            selected_states = st.multiselect("Select States to Compare:", states, default=states[:5])

    # Filter Logic
    df_filtered = df[df['Era'].isin(selected_era)]
    if selected_states:
        df_filtered = df_filtered[df_filtered['state'].isin(selected_states)]

    st.markdown("---")

    # --- SECTION 1: CORE INSIGHT EXPLANATION ---
    st.markdown("### 1. Operational Efficiency Analysis")
    
    st.markdown("""
    <div class='explanation-box'>
        <b>📋 Understanding the Core Insight: The "Batch" vs. "Real-Time" Shift</b><br><br>
        When analyzing this dataset, a casual observer might notice a significant drop in "Total Volumes" starting in September. 
        However, this is not a decline in performance, but a change in <b>Reporting Methodology</b>.<br><br>
        <ul>
            <li><b>Batch Era (Pre-August):</b> Data was uploaded in massive monthly chunks. A single row in the database could represent hundreds of people. This artificially inflated the apparent "volume per entry" but lowered the number of database rows.</li>
            <li><b>Real-Time Era (September+):</b> The system switched to uploading transactions individually or in small daily groups. This results in a lower "volume per entry" but a massive increase in the number of database rows processed.</li>
        </ul>
        <b>The Conclusion:</b> When we look at the "Operational Intensity" (the number of records processed daily), the system has actually <b>increased in efficiency by ~140%</b>. The dashboard below proves this correlation.
    </div>
    """, unsafe_allow_html=True)
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    # KPIs Calculation
    july_vol = df[df['date'].dt.month == 7]['total_enrolment'].sum() / 30
    sept_vol = df[df['date'].dt.month == 9].groupby('date')['total_enrolment'].sum().mean()
    growth = ((sept_vol - july_vol) / july_vol) * 100
    top_state_growth = df_filtered.groupby('state')['total_enrolment'].sum().idxmax()
    
    with col_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="stat-label">Efficiency Growth</div>
            <div class="big-stat">+{growth:.1f}%</div>
            <div style="color: #4CAF50;">▲ True Daily Run Rate (July vs Sept)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="stat-label">Current Processing Rate</div>
            <div class="big-stat">{int(sept_vol):,}</div>
            <div style="color: #b0bec5;">Average Enrolments Per Day (Sept)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="stat-label">Key Volume Driver</div>
            <div class="big-stat" style="font-size: 2em;">{top_state_growth}</div>
            <div style="color: #b0bec5;">Highest Contributor</div>
        </div>
        """, unsafe_allow_html=True)

    # --- VISUALIZATION 1: DUAL AXIS EXPLANATION ---
    st.markdown("#### 📉 Visualization: Volume vs. Operational Intensity")
    st.markdown("""
    **How to Read This Chart:**
    * **Red Area (Left Axis):** This represents the *Total People Enrolled*. Notice the massive spikes in July—these are the "Batches".
    * **Green Line (Right Axis):** This represents the *Number of Database Rows* (Operational Activity). 
    * **The Insight:** Notice that in September, the Red Area gets smaller (no more massive batches), but the Green Line shoots up. This divergence confirms the shift to real-time processing.
    """)

    # --- VISUAL PROOF (Dual Axis) ---
    daily_agg = df.groupby('date').agg({'total_enrolment': 'sum', 'state': 'count'}).rename(columns={'state': 'row_count'})
    
    fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
    fig_combo.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['total_enrolment'], name="Total People Enrolled (Volume)",
                                   line=dict(color='#ff4b4b', width=2), fill='tozeroy', fillcolor='rgba(255, 75, 75, 0.1)'), secondary_y=False)
    fig_combo.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['row_count'], name="Database Rows Created (Activity)",
                                   line=dict(color='#00CC96', width=3, dash='solid')), secondary_y=True)
    
    fig_combo.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=1.1),
        height=500,
        hovermode="x unified"
    )
    st.plotly_chart(fig_combo, use_container_width=True)

    # --- SECTION 2: DEEP DIVE ANALYTICS ---
    st.markdown("### 2. Detailed Data Exploration")
    st.markdown("Click on the tabs below to explore specific dimensions of the data.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Geographic Hierarchy", "🏭 Regional Performance", "🗓️ Weekly Patterns", "🚨 Anomaly Detection"])
    
    with tab1:
        st.markdown("#### 🗺️ Interactive Sunburst Chart")
        st.markdown("""
        **What this shows:** A hierarchical view of enrolment data. 
        * **Inner Circle:** State
        * **Middle Circle:** District
        * **Outer Circle:** Age Group
        
        **How to use:** Click on any "State" slice to zoom in and see its specific Districts. Click the center to zoom back out. This helps identify which specific districts are driving a State's numbers.
        """)
        df_melted = df_filtered.melt(id_vars=['state', 'district'], value_vars=['age_0_5', 'age_5_17', 'age_18_greater'], var_name='Age_Group', value_name='Count')
        sunburst_data = df_melted.groupby(['state', 'district', 'Age_Group'])['Count'].sum().reset_index()
        sunburst_data = sunburst_data[sunburst_data['Count'] > 0]
        
        fig_sun = px.sunburst(sunburst_data, path=['state', 'district', 'Age_Group'], values='Count', color='Count', color_continuous_scale='Viridis')
        fig_sun.update_layout(height=700, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sun, use_container_width=True)

    with tab2:
        st.markdown("#### 🏭 District-Level Performance (Real-Time Era)")
        st.markdown("""
        **What this shows:** The Top 10 Districts based on total enrolment volume, but *filtered only for the Real-Time Era (September onwards)*.
        
        **Why this matters:** Historical data (July) can bias the "Top Districts" list because of the large batch dumps. By looking only at the Real-Time era, we see which centers are *currently* the most active and efficient.
        """)
        
        # Filter for Real-Time Era only for this specific chart
        rt_df = df[df['Era'] == 'Real-Time Era (Sept+)']
        if not rt_df.empty:
            district_growth = rt_df.groupby(['state', 'district'])['total_enrolment'].sum().reset_index()
            district_growth = district_growth.sort_values('total_enrolment', ascending=False).head(10)
            
            fig_bar = px.bar(district_growth, x='total_enrolment', y='district', color='state', orientation='h',
                             text='total_enrolment', title="Top 10 High-Velocity Districts (Sept onwards)")
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Please select 'Real-Time Era' in filters to view this chart.")

    with tab3:
        st.markdown("#### 🗓️ Operational Heatmap")
        st.markdown("""
        **What this shows:** A density map correlating the **Day of the Week** with **States**.
        
        **How to read:** Brighter/Hotter colors indicate higher enrolment activity. 
        * Vertical patterns show which States are busiest.
        * Horizontal patterns show which Days of the Week are busiest.
        * This helps in resource planning (e.g., if Sundays are dead, reduce staff; if Mondays are hot, increase staff).
        """)
        
        # Heatmap: Day of Week vs State
        heatmap_data = df_filtered.groupby(['state', 'DayOfWeek'])['total_enrolment'].sum().reset_index()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig_heat = px.density_heatmap(heatmap_data, x='DayOfWeek', y='state', z='total_enrolment', 
                                      category_orders={'DayOfWeek': days_order}, color_continuous_scale='Hot')
        fig_heat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=600)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab4:
        st.markdown("#### 🚨 Strategic Anomaly: The Meghalaya Outlier")
        st.markdown("""
        **What this shows:** The percentage of enrolments that are **Adults (18+)** for each state.
        
        **The Insight:** Most states have very low adult enrolment (<5%) because most adults are already enrolled; they are primarily enrolling children.
        **The Anomaly:** **Meghalaya** is a massive outlier with **~32%** adult enrolment.
        
        **Recommendation:** This indicates a specific backlog or migration pattern in Meghalaya. Resources sent there should be specialized for **Adult Biometrics**, whereas the rest of India needs **Child Enrolment Kits**.
        """)
        
        state_stats = df.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater', 'total_enrolment']].sum()
        state_stats['pct_18_plus'] = (state_stats['age_18_greater'] / state_stats['total_enrolment']) * 100
        state_stats = state_stats.sort_values('pct_18_plus', ascending=False).head(10).reset_index()
        
        colors = ['#ff4b4b' if x == 'Meghalaya' else '#2c5364' for x in state_stats['state']]
        
        fig_ano = px.bar(state_stats, x='state', y='pct_18_plus', text='pct_18_plus', title="Percentage of Adult Enrolments (18+)")
        fig_ano.update_traces(marker_color=colors, texttemplate='%{text:.1f}%')
        fig_ano.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_ano, use_container_width=True)

else:
    st.error("Data files not found. Please upload the CSV files.")