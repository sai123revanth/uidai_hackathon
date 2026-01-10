import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Aadhar Lifecycle Strategy Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# Custom CSS for styling - Upgraded for Navy Blue Gradient Theme
st.markdown("""
<style>
    /* Main App Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #000428 0%, #004e92 100%);
        color: #ffffff;
    }

    /* Target specific container for background to ensure full coverage */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #020c1b 0%, #061a33 100%);
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05); 
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 191, 255, 0.4);
    }

    .metric-title {
        color: #8892b0; 
        font-size: 14px;
        margin-bottom: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    .metric-value {
        font-size: 38px;
        font-weight: 800;
        line-height: 1.2;
        background: linear-gradient(to right, #ffffff, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Gradient Font Colors for Highlighting */
    .highlight-red { color: #ff4d4d !important; text-shadow: 0 0 10px rgba(255, 77, 77, 0.3); }
    .highlight-green { color: #00ff88 !important; text-shadow: 0 0 10px rgba(0, 255, 136, 0.3); }
    .highlight-blue { color: #00d2ff !important; text-shadow: 0 0 10px rgba(0, 210, 255, 0.3); }
    .highlight-orange { color: #ff9f43 !important; text-shadow: 0 0 10px rgba(255, 159, 67, 0.3); }
    
    /* Box Styles */
    .guide-box {
        background-color: rgba(0, 210, 255, 0.03);
        border-left: 4px solid #00d2ff;
        padding: 20px;
        margin: 10px 0 25px 0;
        font-size: 14px;
        line-height: 1.6;
        border-radius: 0 10px 10px 0;
        color: #ccd6f6;
    }

    .insight-header {
        color: #00d2ff;
        font-weight: bold;
        margin-bottom: 10px;
        display: block;
        font-size: 18px;
        letter-spacing: 0.5px;
    }

    /* Navigation Bar styling for Filters */
    .nav-filter-container {
        background-color: rgba(255, 255, 255, 0.02);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* General Typography Fixes for Dark Theme */
    h1, h2, h3, h4, h5, p, span, li {
        color: #e6f1ff;
    }
    
    .stMarkdown div p {
        color: #ccd6f6;
    }

    /* Adjusting Streamlit internal elements */
    .stSelectbox label, .stMultiSelect label, .stDateInput label {
        color: #00d2ff !important;
        font-weight: 600 !important;
    }
    
    /* Dataframe visibility */
    .stDataFrame {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading Functions ---

@st.cache_data
def load_and_process_data():
    """
    Loads all data files (including newly uploaded ones), standardizes columns, 
    and merges them into a master dataframe aggregated by Date, State, and District.
    """
    # Comprehensive File lists including all provided data segments
    enrolment_files = [
        "api_data_aadhar_enrolment_0_500000.csv",
        "api_data_aadhar_enrolment_500000_1000000.csv",
        "api_data_aadhar_enrolment_1000000_1006029.csv"
    ]
    demo_files = [
        "api_data_aadhar_demographic_0_500000.csv",
        "api_data_aadhar_demographic_500000_1000000.csv",
        "api_data_aadhar_demographic_1000000_1500000.csv",
        "api_data_aadhar_demographic_1500000_2000000.csv",
        "api_data_aadhar_demographic_2000000_2071700.csv"
    ]
    bio_files = [
        "api_data_aadhar_biometric_0_500000.csv",
        "api_data_aadhar_biometric_500000_1000000.csv",
        "api_data_aadhar_biometric_1000000_1500000.csv",
        "api_data_aadhar_biometric_1500000_1861108.csv"
    ]

    # Helper to read and concat multiple CSVs for a category
    def read_files(file_list, type_tag):
        df_list = []
        for f in file_list:
            if os.path.exists(f):
                try:
                    # Low memory set to False to handle large mixed data chunks safely
                    df_chunk = pd.read_csv(f, low_memory=False)
                    df_list.append(df_chunk)
                except Exception as e:
                    st.error(f"Error reading {f}: {e}")
        
        if not df_list:
            return pd.DataFrame()
        
        full_df = pd.concat(df_list, ignore_index=True)
        return full_df

    # 1. Process Enrolment Data
    df_enrol = read_files(enrolment_files, "Enrolment")
    if not df_enrol.empty:
        df_enrol.columns = df_enrol.columns.str.strip()
        cols = ['age_0_5', 'age_5_17', 'age_18_greater']
        for c in cols:
            if c in df_enrol.columns:
                df_enrol[c] = pd.to_numeric(df_enrol[c], errors='coerce').fillna(0)
        df_enrol['New_Enrolments'] = df_enrol['age_0_5'] + df_enrol['age_5_17'] + df_enrol['age_18_greater']
        df_enrol_agg = df_enrol.groupby(['date', 'state', 'district'])[['New_Enrolments', 'age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
    else:
        df_enrol_agg = pd.DataFrame(columns=['date', 'state', 'district', 'New_Enrolments'])

    # 2. Process Demographic Update Data
    df_demo = read_files(demo_files, "Demographic")
    if not df_demo.empty:
        df_demo.columns = df_demo.columns.str.strip()
        cols = ['demo_age_5_17', 'demo_age_17_'] 
        for c in cols:
            if c in df_demo.columns:
                df_demo[c] = pd.to_numeric(df_demo[c], errors='coerce').fillna(0)
        df_demo['Demographic_Updates'] = df_demo['demo_age_5_17'] + df_demo['demo_age_17_']
        df_demo_agg = df_demo.groupby(['date', 'state', 'district'])[['Demographic_Updates', 'demo_age_5_17', 'demo_age_17_']].sum().reset_index()
    else:
        df_demo_agg = pd.DataFrame(columns=['date', 'state', 'district', 'Demographic_Updates'])

    # 3. Process Biometric Update Data
    df_bio = read_files(bio_files, "Biometric")
    if not df_bio.empty:
        df_bio.columns = df_bio.columns.str.strip()
        cols = ['bio_age_5_17', 'bio_age_17_']
        for c in cols:
             if c in df_bio.columns:
                df_bio[c] = pd.to_numeric(df_bio[c], errors='coerce').fillna(0)
        df_bio['Biometric_Updates'] = df_bio['bio_age_5_17'] + df_bio['bio_age_17_']
        df_bio_agg = df_bio.groupby(['date', 'state', 'district'])[['Biometric_Updates', 'bio_age_5_17', 'bio_age_17_']].sum().reset_index()
    else:
        df_bio_agg = pd.DataFrame(columns=['date', 'state', 'district', 'Biometric_Updates'])

    # 4. Merge All Data (Outer join ensures no date/state/district data is lost)
    df_master = pd.merge(df_enrol_agg, df_demo_agg, on=['date', 'state', 'district'], how='outer').fillna(0)
    df_master = pd.merge(df_master, df_bio_agg, on=['date', 'state', 'district'], how='outer').fillna(0)

    # 5. Final Formatting & Feature Engineering
    df_master['date'] = pd.to_datetime(df_master['date'], dayfirst=True, errors='coerce')
    df_master['Total_Updates'] = df_master['Demographic_Updates'] + df_master['Biometric_Updates']
    df_master = df_master.dropna(subset=['date'])
    df_master['state'] = df_master['state'].str.title().str.strip()
    df_master['district'] = df_master['district'].str.title().str.strip()
    
    # Seasonality features
    df_master['day_of_week'] = df_master['date'].dt.day_name()
    df_master['month'] = df_master['date'].dt.month_name()
    df_master['month_num'] = df_master['date'].dt.month
    df_master['year'] = df_master['date'].dt.year

    return df_master

# Execute Data Load
try:
    df = load_and_process_data()
except Exception as e:
    st.error(f"Critical Error Loading Consolidated Data: {e}")
    st.stop()

if df.empty:
    st.warning("No data found. Please ensure all CSV files are uploaded and named correctly.")
    st.stop()

# --- Dashboard Header Section ---

st.title("🇮🇳 Aadhar Strategic Pivot Dashboard")
st.subheader("Transitioning from 'New Acquisition' to 'Continuous Maintenance'")

st.markdown("""
<div style='background: linear-gradient(90deg, rgba(0, 210, 255, 0.15) 0%, rgba(58, 123, 213, 0.05) 100%); padding: 20px; border-radius: 12px; border-left: 6px solid #00d2ff; margin-bottom: 25px;'>
    <h4 style="margin-top:0; color: #00d2ff; letter-spacing: 0.5px;">💡 Executive Context (Multi-File Consolidated)</h4>
    <p style="margin:0; color: #ccd6f6;">India's identity infrastructure is moving from an <strong>Enrolment Phase</strong> (getting everyone an ID) to a 
    <strong>Lifecycle Phase</strong> (keeping IDs accurate). This dashboard tracks that transition using the full data workload. 
    A high "Update Ratio" means the region is fully saturated and requires maintenance infrastructure, not enrolment kits.</p>
</div>
""", unsafe_allow_html=True)

# --- Navigation Bar Filter Analysis ---
st.markdown("#### 🎛️ Filter Analysis & Controls")
with st.container():
    col_nav1, col_nav2, col_nav3 = st.columns([1.2, 1, 1])
    
    with col_nav1:
        min_date = df['date'].min()
        max_date = df['date'].max()
        if not pd.isnull(min_date) and not pd.isnull(max_date):
            selected_range = st.date_input(
                "Select Date Range",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            if len(selected_range) == 2:
                start_date, end_date = selected_range
            else:
                start_date, end_date = min_date, max_date
        else:
            start_date, end_date = min_date, max_date

    with col_nav2:
        # User selection for analysis granularity
        all_states = sorted(df['state'].unique())
        selected_states = st.multiselect("Select State(s)", all_states, default=[])

    with col_nav3:
        if selected_states:
            filtered_districts_list = sorted(df[df['state'].isin(selected_states)]['district'].unique())
        else:
            filtered_districts_list = sorted(df['district'].unique())
        selected_districts = st.multiselect("Select District(s)", filtered_districts_list)

# --- Apply Filter Mask ---
mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
if selected_states:
    mask = mask & (df['state'].isin(selected_states))
if selected_districts:
    mask = mask & (df['district'].isin(selected_districts))

filtered_df = df.loc[mask]

# --- KPI Section ---
total_enrolments = filtered_df['New_Enrolments'].sum()
total_demo_updates = filtered_df['Demographic_Updates'].sum()
total_bio_updates = filtered_df['Biometric_Updates'].sum()
total_updates = total_demo_updates + total_bio_updates
ratio = total_updates / total_enrolments if total_enrolments > 0 else total_updates

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">New Enrolments</div><div class="metric-value highlight-green">{total_enrolments:,.0f}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Demographic Updates</div><div class="metric-value highlight-blue">{total_demo_updates:,.0f}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Biometric Updates</div><div class="metric-value highlight-orange">{total_bio_updates:,.0f}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Update-to-Enrol Ratio</div><div class="metric-value highlight-red">{ratio:.1f}x</div></div>', unsafe_allow_html=True)

# KPI Detailed Explanation
st.markdown("""
<div class="guide-box">
    <span class="insight-header">📋 How to Read the KPIs:</span>
    <ul style="color: #ccd6f6;">
        <li><strong>New Enrolments:</strong> Number of unique Aadhaar IDs generated. A decline here isn't bad—it indicates high saturation.</li>
        <li><strong>Updates (Demo/Bio):</strong> Volume of changes to existing IDs. This is your primary workload now.</li>
        <li><strong>Update-to-Enrol Ratio:</strong> The critical "Saturation Metric." 
            <ul>
                <li><strong>< 5x:</strong> Growth Phase. Prioritize onboarding new users.</li>
                <li><strong>5x - 14x:</strong> Transition Phase. Balance onboarding and maintenance.</li>
                <li><strong>> 14x:</strong> Mature Phase. Infrastructure should be 90% service-oriented.</li>
            </ul>
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- Row 1: Comparison and Growth ---
st.markdown("---")
col_growth_1, col_growth_2 = st.columns([1, 1])

with col_growth_1:
    st.subheader("📊 Category Radar Comparison")
    categories = ['Enrolment', 'Demographic', 'Biometric']
    values = [total_enrolments, total_demo_updates, total_bio_updates]
    max_val = max(values) if max(values) > 0 else 1
    norm_values = [v/max_val for v in values]
    
    fig_radar = go.Figure(data=go.Scatterpolar(r=norm_values, theta=categories, fill='toself', line_color='#00d2ff'))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)"
        ), 
        showlegend=False, 
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ccd6f6"),
        template="plotly_dark"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    with st.expander("🔍 Deep Detail: Radar Analysis"):
        st.write("""
        This chart visualizes the "Functional Identity" of your operations.
        - **Pull toward Enrolment:** Indicates the region is still in an "Acquisition" mode. You need more mobile camps and field staff.
        - **Pull toward Demographic:** Suggests high residential mobility or data correction needs (names, addresses). This can often be handled via online portals or kiosks.
        - **Pull toward Biometric:** This is the most hardware-intensive need. It implies users are updating photos or fingerprints (mandatory at ages 5 and 15). This requires physical centers with high-end scanners.
        """)

with col_growth_2:
    st.subheader("📈 Cumulative Growth Comparison")
    daily_growth = filtered_df.groupby('date')[['New_Enrolments', 'Total_Updates']].sum().cumsum().reset_index()
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=daily_growth['date'], y=daily_growth['New_Enrolments'], name='New Enrolments', fill='tozeroy', line_color="#00ff88"))
    fig_cum.add_trace(go.Scatter(x=daily_growth['date'], y=daily_growth['Total_Updates'], name='Total Updates', fill='tonexty', line_color="#00d2ff"))
    fig_cum.update_layout(
        title="Volume Progression", 
        hovermode="x unified", 
        height=400, 
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ccd6f6")
    )
    st.plotly_chart(fig_cum, use_container_width=True)
    
    with st.expander("🔍 Deep Detail: Growth Velocity"):
        st.write("""
        This chart shows the "Area Under the Curve," representing the total historical workload.
        - **The Gap:** The vertical distance between 'Total Updates' and 'New Enrolments' is your **Cumulative Maintenance Burden**. 
        - **Slope Analysis:** If the slope of the 'Updates' line is getting steeper over time, it means your existing user base is becoming more active, even if new enrolments are flat.
        - **Strategic Use:** Use this to project future staffing needs based on the historical acceleration of the update volume.
        """)

# --- Row 2: Seasonality Heatmap & Monthly Trends ---
st.markdown("---")
col_seasonal_1, col_seasonal_2 = st.columns(2)

with col_seasonal_1:
    st.subheader("🗓️ Weekly Operational Peaks")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = filtered_df.groupby('day_of_week')[['Total_Updates']].mean().reindex(day_order).reset_index()
    fig_heat = px.bar(heatmap_data, x='day_of_week', y='Total_Updates', color='Total_Updates',
                     color_continuous_scale='Blues', labels={'Total_Updates': 'Avg Daily Load'})
    fig_heat.update_layout(
        height=400, 
        template="plotly_dark", 
        xaxis_title="",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ccd6f6")
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("""
    **Insight Detail:** This identifies **Staffing Requirements**. If Saturday/Sunday bars are higher, it indicates that users prefer weekend updates. Operationally, this means you should negotiate 'Floating Offs' for your operators to ensure 7-day coverage without increasing headcount.
    """)

with col_seasonal_2:
    st.subheader("📅 Monthly Load Cycles")
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data = filtered_df.groupby(['month', 'month_num'])[['Total_Updates', 'New_Enrolments']].mean().sort_values('month_num').reset_index()
    fig_month = go.Figure()
    fig_month.add_trace(go.Bar(x=monthly_data['month'], y=monthly_data['Total_Updates'], name='Avg Updates', marker_color='#00d2ff'))
    fig_month.add_trace(go.Bar(x=monthly_data['month'], y=monthly_data['New_Enrolments'], name='Avg Enrolments', marker_color='#00ff88'))
    fig_month.update_layout(
        height=400, 
        barmode='group', 
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ccd6f6")
    )
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("""
    **Insight Detail:** This identifies **Annual Periodicity**. Peaks in specific months (e.g., June/July) often correlate with school admissions where Aadhaar updates are mandatory. Use this to plan for seasonal 'Peak Capacity' surges.
    """)

# --- Row 3: Intensity Scatter & Regional Hotspots ---
st.markdown("---")
st.subheader("🎯 Strategic Intensity & Regional Hotspots")
col_scatter, col_geo_list = st.columns([1.5, 1])

with col_scatter:
    # Analyzing the relationship between Demo and Bio updates at a granular level
    geo_scatter = filtered_df.groupby('state' if not selected_districts else 'district')[['Demographic_Updates', 'Biometric_Updates', 'New_Enrolments']].sum().reset_index()
    geo_scatter['Size'] = geo_scatter['New_Enrolments'].apply(lambda x: np.log(x + 1) * 5) # Scale for bubble
    
    fig_scatter = px.scatter(
        geo_scatter, x='Demographic_Updates', y='Biometric_Updates', 
        text='state' if not selected_districts else 'district',
        size='New_Enrolments', color='Demographic_Updates',
        title="Update Type Intensity by Region",
        labels={'Demographic_Updates': 'Demographic Volume', 'Biometric_Updates': 'Biometric Volume'},
        template="plotly_dark",
        color_continuous_scale="Viridis"
    )
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ccd6f6")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    with st.expander("🔍 Deep Detail: Intensity Scatter Analysis"):
        st.write("""
        This 4-Quadrant analysis identifies the **Profile** of each region:
        - **High Demo / High Bio (Top Right):** Major Hubs. These regions need large, permanent "Mega Centers" with specialized counters for both types.
        - **High Demo / Low Bio (Bottom Right):** "Digital First" regions. These areas might benefit from more self-service kiosks as demographic updates are easier to digitize.
        - **Low Demo / High Bio (Top Left):** "Hardware Deficit" regions. These are likely rural or high-churn areas where mandatory biometric updates (MBU) are the main driver. They need more mobile biometric kits.
        - **Bubble Size:** Represents new enrolments. A small bubble far in the top-right is the definition of a "Saturated Service-Only" district.
        """)

with col_geo_list:
    st.markdown("#### 🏆 Performance Benchmarking")
    geo_group = filtered_df.groupby('state' if not selected_districts else 'district')[['New_Enrolments', 'Total_Updates']].sum().reset_index()
    geo_group['Update_Ratio'] = geo_group['Total_Updates'] / geo_group['New_Enrolments'].replace(0, 1)
    
    # Sort and highlight
    top_efficiency = geo_group.sort_values('Update_Ratio', ascending=False).head(10)
    st.write("**Top 10 Saturated Regions (High Ratio):**")
    st.dataframe(top_efficiency[['state' if not selected_districts else 'district', 'Update_Ratio']].style.format({"Update_Ratio": "{:.2f}x"}).background_gradient(cmap='Blues'), use_container_width=True)
    
    st.markdown("""
    **What this list tells you:** These are the regions where the old "Enrolment First" KPIs are no longer applicable. Management should judge these centers based on **Customer Waiting Time (CWT)** and **Update Success Rate** rather than "IDs generated."
    """)

# --- Row 4: Demographic Breakdowns & Lifecycle Segments ---
st.markdown("---")
st.subheader("👥 User Segmentation & Lifecycle Analysis")
col_age_1, col_age_2 = st.columns(2)

with col_age_1:
    st.markdown("#### New Enrolment Funnel")
    enrol_age = pd.DataFrame({
        'Age Group': ['Infant (0-5)', 'Youth (5-17)', 'Adult (18+)'],
        'Count': [filtered_df['age_0_5'].sum(), filtered_df['age_5_17'].sum(), filtered_df['age_18_greater'].sum()]
    })
    fig_age_enrol = px.pie(enrol_age, values='Count', names='Age Group', hole=0.5, 
                           color_discrete_sequence=px.colors.sequential.GnBu_r)
    fig_age_enrol.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ccd6f6")
    )
    st.plotly_chart(fig_age_enrol, use_container_width=True)
    
    st.markdown("""
    **Strategic Insight:** In a perfect system, 100% of this chart should be **0-5 years**. If the **18+ years** slice is large, it indicates a significant "Last Mile" failure where adults are only now getting IDs—likely due to mandatory linkings for welfare or banking.
    """)

with col_age_2:
    st.markdown("#### Operational Complexity Split")
    update_type = pd.DataFrame({
        'Type': ['Demographic (Low Complexity)', 'Biometric (High Complexity)'],
        'Count': [total_demo_updates, total_bio_updates]
    })
    fig_upd_type = px.pie(update_type, values='Count', names='Type', hole=0.5, 
                           color_discrete_sequence=px.colors.sequential.Blues_r)
    fig_upd_type.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#ccd6f6")
    )
    st.plotly_chart(fig_upd_type, use_container_width=True)

    st.markdown("""
    **Strategic Insight:** Biometric updates take on average **3x longer** to process than demographic ones. If the "High Complexity" slice is growing, your "Throughput" (number of people per hour) will naturally decrease. Management must not penalize centers for lower volumes if their complexity mix is shifting toward Biometrics.
    """)

# --- Row 5: Detailed Data Inspection ---
st.markdown("---")
with st.expander("🔍 Deep Dive: Master Transaction Log (Auditable Data)"):
    st.markdown("""
    **How to use this table:** 1. Click headers to sort by date or volume. 
    2. Use this for 'Spot Audits' of specific districts.
    3. Look for 'Zero Enrolment' days—these are the days a district effectively transitioned into a pure Service Center.
    """)
    st.dataframe(filtered_df.sort_values(['date', 'Total_Updates'], ascending=[False, False]), use_container_width=True)

# --- Strategic Recommendations (Dynamic based on data) ---
st.markdown("---")
st.markdown("### 🎯 Final Strategic Directives")

# Dynamic logic for final advice
avg_ratio = ratio
if avg_ratio > 10:
    status_label = "MATURE SERVICE ECOSYSTEM"
    status_color = "#ff4d4d"
else:
    status_label = "HYBRID GROWTH ECOSYSTEM"
    status_color = "#00d2ff"

st.markdown(f"""
<div style='border: 2px solid {status_color}; background: rgba(0,0,0,0.2); padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 0 20px {status_color}33;'>
    <h2 style='color: {status_color}; margin: 0; letter-spacing: 2px;'>CURRENT STRATEGY: {status_label}</h2>
    <p style='color: #ccd6f6; font-size: 1.1em;'>Based on the calculated Update-to-Enrol ratio of <strong style='color:#ffffff;'>{avg_ratio:.1f}x</strong></p>
</div>
""", unsafe_allow_html=True)

rec_col1, rec_col2 = st.columns(2)
with rec_col1:
    st.markdown(f"""
    <div style="background: rgba(0, 255, 136, 0.05); padding: 20px; border-radius: 10px; border-left: 5px solid #00ff88;">
        <h4 style="color: #00ff88; margin-top: 0;">🛠️ Operational Pivot</h4>
        <ul style="color: #ccd6f6;">
            <li><strong>Update Center Conversion:</strong> In all regions showing > 15x ratio, convert Enrolment Kits to 'Update Kiosks'.</li>
            <li><strong>Skill Training:</strong> Shift operator training from "How to enroll" to "How to verify biometric consistency" to reduce update rejection rates.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with rec_col2:
    st.markdown(f"""
    <div style="background: rgba(255, 159, 67, 0.05); padding: 20px; border-radius: 10px; border-left: 5px solid #ff9f43;">
        <h4 style="color: #ff9f43; margin-top: 0;">📡 Infrastructure Deployment</h4>
        <ul style="color: #ccd6f6;">
            <li><strong>Hardware Refresh:</strong> Prioritize Biometric Scanner upgrades in the 'Low Demo / High Bio' quadrants identified in the scatter plot.</li>
            <li><strong>Mobile Strategy:</strong> If adult enrolments (18+) are high in specific districts, deploy short-term 'Saturation Camps' to close the gap quickly and move to maintenance.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer spacing
st.markdown("<br><br>", unsafe_allow_html=True)