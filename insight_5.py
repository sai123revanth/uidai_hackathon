import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Aadhar Operational Friction Analytics",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Styling for "Advanced" Navy Blue Theme ---
st.markdown("""
<style>
    /* Global Background: Navy Blue Gradient */
    .stApp {
        background: linear-gradient(135deg, #000428 0%, #004e92 100%);
        background-attachment: fixed;
        color: #FFFFFF;
    }

    /* Text Coloring for High Contrast */
    h1, h2, h3, h4, h5, h6, strong {
        color: #FFFFFF !important;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.5);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    p, label, .stMarkdown, .stMultiSelect label, .stExpander {
        color: #E0E0E0 !important;
    }
    
    /* Metric Cards - Glassmorphism */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #B0B0B0 !important;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #00F2FF !important; /* Cyan highlight for numbers */
        font-weight: 700;
    }
    div[data-testid="stMetricDelta"] {
        color: #FFD700 !important; /* Gold for deltas */
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 4, 40, 0.8);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Recommendation Box Update */
    .recommendation-box {
        padding: 25px;
        background: rgba(26, 188, 156, 0.15);
        border: 1px solid #1abc9c;
        border-left: 6px solid #1abc9c;
        border-radius: 8px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .recommendation-box h3 {
        color: #1abc9c !important;
        margin-top: 0;
        font-size: 1.5rem;
    }
    .recommendation-box p {
        color: #E0E0E0 !important;
        font-size: 1.1rem;
    }
    
    /* Method Box Styling for visibility without expander */
    .methodology-box {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Data Loading & Caching ---
@st.cache_data
def load_and_prep_data():
    try:
        # Load Biometric Data
        bio_files = [
            'api_data_aadhar_biometric_0_500000.csv',
            'api_data_aadhar_biometric_500000_1000000.csv'
        ]
        df_bio = pd.concat([pd.read_csv(f) for f in bio_files], ignore_index=True)
        
        # Load Demographic Data
        demo_files = [
            'api_data_aadhar_demographic_1000000_1500000.csv',
            'api_data_aadhar_demographic_1500000_2000000.csv'
        ]
        df_demo = pd.concat([pd.read_csv(f) for f in demo_files], ignore_index=True)
        
        # Load Enrolment Data (for context/population proxy)
        enrol_files = [
            'api_data_aadhar_enrolment_0_500000.csv',
            'api_data_aadhar_enrolment_500000_1000000.csv'
        ]
        df_enrol = pd.concat([pd.read_csv(f) for f in enrol_files], ignore_index=True)

        return df_bio, df_demo, df_enrol

    except FileNotFoundError as e:
        st.error(f"File not found: {e}. Please ensure all CSV files are in the same directory.")
        return None, None, None

def process_data(df_bio, df_demo, df_enrol):
    # Standardize column names (handling potential cutoffs or extra spaces)
    df_bio.columns = [c.strip().lower() for c in df_bio.columns]
    df_demo.columns = [c.strip().lower() for c in df_demo.columns]
    df_enrol.columns = [c.strip().lower() for c in df_enrol.columns]
    
    # Identify value columns dynamically (assuming last 2 columns are counts)
    bio_val_cols = [c for c in df_bio.columns if 'bio_age' in c]
    demo_val_cols = [c for c in df_demo.columns if 'demo_age' in c]
    enrol_val_cols = [c for c in df_enrol.columns if 'age_' in c]
    
    # Aggregation: Group by State and District
    # We aggregate dates to get a total operational view
    bio_agg = df_bio.groupby(['state', 'district'])[bio_val_cols].sum().reset_index()
    bio_agg['Total_Biometric_Updates'] = bio_agg[bio_val_cols].sum(axis=1)
    
    demo_agg = df_demo.groupby(['state', 'district'])[demo_val_cols].sum().reset_index()
    demo_agg['Total_Demographic_Updates'] = demo_agg[demo_val_cols].sum(axis=1)

    enrol_agg = df_enrol.groupby(['state', 'district'])[enrol_val_cols].sum().reset_index()
    enrol_agg['Total_Enrolments'] = enrol_agg[enrol_val_cols].sum(axis=1)
    
    # Merge datasets
    df_master = pd.merge(bio_agg, demo_agg, on=['state', 'district'], how='outer')
    df_master = pd.merge(df_master, enrol_agg[['state', 'district', 'Total_Enrolments']], on=['state', 'district'], how='left').fillna(0)
    
    # --- CALCULATE THE FRICTION INDEX ---
    # Formula: Bio Updates / (Demo Updates + 1)
    # Logic: High Bio + Low Demo = High Friction (Fingerprint failures)
    df_master['Friction_Index'] = df_master['Total_Biometric_Updates'] / (df_master['Total_Demographic_Updates'] + 1)
    
    # Clean up state/district names (capitalize)
    df_master['state'] = df_master['state'].str.title()
    df_master['district'] = df_master['district'].str.title()
    
    return df_master

# --- Main App Execution ---
df_bio_raw, df_demo_raw, df_enrol_raw = load_and_prep_data()

if df_bio_raw is not None:
    df_analysis = process_data(df_bio_raw, df_demo_raw, df_enrol_raw)
    
    # --- Header Section ---
    st.title("🇮🇳 Operational Intelligence: Friction Analytics")

    # --- Top Navigation / Filter Bar ---
    # Moved from sidebar to main area for better visibility
    with st.container():
        st.markdown("### 🔍 Advanced Filters")
        f_col1, f_col2 = st.columns(2)
        
        all_states = sorted(df_analysis['state'].unique())
        
        with f_col1:
            # Removed default selection to show all data by default
            selected_states = st.multiselect("Select State(s)", all_states)
        
        # Filter Logic
        if selected_states:
            df_filtered = df_analysis[df_analysis['state'].isin(selected_states)]
        else:
            df_filtered = df_analysis # Show all if none selected

        # District Filter (Dependent)
        all_districts = sorted(df_filtered['district'].unique())
        
        with f_col2:
            selected_districts = st.multiselect("Select District(s)", all_districts)
        
        if selected_districts:
            df_filtered = df_filtered[df_filtered['district'].isin(selected_districts)]
        
    # --- Add Friction Category for coloring in multiple charts ---
    df_filtered['Status'] = np.where(df_filtered['Friction_Index'] > 3, 'Critical Friction', 
                            np.where(df_filtered['Friction_Index'] > 1, 'Moderate', 'Normal'))
    
    # --- Methodology Section (Now Visible, No Expander) ---
    st.markdown('<div class="methodology-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Analytics Methodology & Module Explanation")
    st.markdown("""
    **About this Module:**
    This dashboard utilizes **Biometric vs. Demographic disparity logic** to identify hardware failures or population-specific authentication issues (e.g., worn fingerprints).
    
    **Statistical Methods Applied:**
    1.  **Univariate Analysis:** We examine the distribution of the `Friction Index` (Histogram) to understand if high friction is a systemic issue or restricted to outliers.
    2.  **Bivariate Analysis:** We compare `Total Biometric Updates` against `Total Demographic Updates` (Scatter Plot) to isolate anomalies from standard operational noise.
    3.  **Trivariate Analysis:** We correlate `Enrolment Volume`, `Update Volume`, and `Friction Index` to see if legacy data quality (historical enrolments) is driving current failures.
    
    **The Core Metric:**
    $Friction Index = \\frac{Biometric Updates}{Demographic Updates + 1}$
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- KPI Row ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_bio = df_filtered['Total_Biometric_Updates'].sum()
    total_demo = df_filtered['Total_Demographic_Updates'].sum()
    avg_friction = df_filtered['Friction_Index'].mean()
    high_friction_count = df_filtered[df_filtered['Friction_Index'] > 2].shape[0] # Threshold assumption
    
    kpi1.metric("Total Biometric Updates", f"{total_bio:,.0f}", delta="Operational Load")
    kpi2.metric("Total Demographic Updates", f"{total_demo:,.0f}")
    kpi3.metric("Avg Friction Index", f"{avg_friction:.2f}", delta="Efficiency Metric", delta_color="off")
    kpi4.metric("Critical Districts", f"{high_friction_count}", delta="Requires Attention", delta_color="inverse")

    # --- Section 1: Univariate & Bivariate Distributions ---
    st.header("1. Distribution & Outlier Detection")
    st.markdown("Analyzing the spread of friction data using **Univariate (Histogram)** and **Bivariate (Bar)** techniques.")
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("Distribution of Friction Index (Univariate)")
        st.markdown("""
        **Analysis Type:** Univariate Probability Density.
        **Insight:** A "Long Tail" to the right indicates that while most districts are healthy (Low Friction), a few specific districts are suffering severe failures.
        """)
        fig_hist = px.histogram(
            df_filtered, 
            x="Friction_Index",
            nbins=30,
            color="Status",
            title="Frequency Distribution of Friction Scores",
            color_discrete_map={'Critical Friction': '#FF4B4B', 'Moderate': '#FFA500', 'Normal': '#00CC96'},
            template='plotly_dark'
        )
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_hist, use_container_width=True)

    with row1_col2:
        st.subheader("State-wise Friction Comparison (Bivariate)")
        st.markdown("""
        **Analysis Type:** Bivariate (Categorical vs Numerical).
        **Insight:** Aggregating friction by state helps identify if the problem is **Policy/Infrastructure Level** (High State Avg) or **District Specific** (Low State Avg, High Variance).
        """)
        state_avg = df_filtered.groupby('state')['Friction_Index'].mean().reset_index().sort_values('Friction_Index', ascending=False)
        fig_state = px.bar(
            state_avg,
            x='state',
            y='Friction_Index',
            color='Friction_Index',
            color_continuous_scale='Reds',
            title="Average Friction Index by State",
            template='plotly_dark'
        )
        fig_state.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_state, use_container_width=True)

    # --- Section 2: Geospatial & Operational Hierarchy ---
    st.header("2. The Friction Heatmap (Multivariate)")
    st.markdown("Visualizing **Operational Stress** hierarchically. **Size** = Workload Volume, **Color** = Friction Severity.")

    # We use a Treemap because it handles hierarchical data (State > District) better than a map without shapefiles
    # FIX: Added range_color=[0, 5] to prevent outliers (e.g., Index=100) from washing out the color scale. 
    # Now, anything above 5.0 is maximum Red.
    fig_treemap = px.treemap(
        df_filtered,
        path=[px.Constant("India"), 'state', 'district'],
        values='Total_Biometric_Updates',
        color='Friction_Index',
        color_continuous_scale='RdYlGn_r', # Red is High Friction (Bad), Green is Low (Good)
        range_color=[0, 5], # Cap the scale at 5.0 to handle outliers
        title='Friction Heatmap: Red Zones = High Bio Updates / Low Demo Updates',
        hover_data=['Total_Demographic_Updates', 'Friction_Index', 'Total_Enrolments'],
        template='plotly_dark' 
    )
    fig_treemap.update_layout(
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

    # --- Section 3: Deep Dive Correlation Analytics ---
    st.header("3. Correlation & Causality Analysis")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Operational Disparity (Bivariate)")
        st.markdown("""
        **Analysis Type:** Bivariate Scatter (Log Scale).
        **Insight:** Districts above the diagonal line have disproportionately high biometric updates compared to demographic changes, flagging potential **Hardware Issues** or **Worn Fingerprints**.
        """)
        
        fig_scatter = px.scatter(
            df_filtered,
            x='Total_Demographic_Updates',
            y='Total_Biometric_Updates',
            color='Status',
            size='Total_Biometric_Updates',
            hover_name='district',
            log_x=True, log_y=True, # Log scale handles the massive variance in district sizes
            color_discrete_map={'Critical Friction': '#FF4B4B', 'Moderate': '#FFA500', 'Normal': '#00CC96'},
            title="Biometric vs Demographic Volume (Log Scale)",
            template='plotly_dark'
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("Historical Impact (Trivariate)")
        st.markdown("""
        **Analysis Type:** Trivariate (Enrolment vs Friction vs Size).
        **Insight:** Does high historical enrolment lead to current friction?
        * **X-Axis:** Total Enrolments (History)
        * **Y-Axis:** Friction Index (Current Quality)
        * **Bubble Size:** Biometric Update Volume
        """)
        
        fig_bubble = px.scatter(
            df_filtered,
            x='Total_Enrolments',
            y='Friction_Index',
            color='Status',
            size='Total_Biometric_Updates',
            hover_name='district',
            title="Legacy Impact: Enrolment History vs Current Friction",
            color_discrete_map={'Critical Friction': '#FF4B4B', 'Moderate': '#FFA500', 'Normal': '#00CC96'},
            template='plotly_dark'
        )
        fig_bubble.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    # --- Section 4: Leaderboards & Actions ---
    st.subheader("Leaderboard: Critical Zones")
    st.markdown("_Districts with the highest ratio of biometric failures._")
    
    top_friction = df_filtered.sort_values(by='Friction_Index', ascending=False).head(15)
    
    fig_bar = px.bar(
        top_friction,
        x='Friction_Index',
        y='district',
        orientation='h',
        color='Friction_Index',
        color_continuous_scale='Reds',
        text_auto='.1f',
        title="Highest Friction Indices",
        template='plotly_dark'
    )
    fig_bar.update_layout(
        yaxis={'categoryorder':'total ascending'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Recommendations Module ---
    st.markdown("---")
    st.header("🛠️ Operational Recommendations")
    
    # Logic for recommendations
    # Threshold: Friction Index > 2.0 (Twice as many Bio updates as Demo updates)
    critical_districts = df_filtered[df_filtered['Friction_Index'] > 2.0].sort_values(by='Total_Biometric_Updates', ascending=False)
    
    st.markdown(f"""
    <div class="recommendation-box">
        <h3>🚀 Action Plan: Deploy Iris Scanners</h3>
        <p>Based on the analytics, <b>{len(critical_districts)} districts</b> have been identified as "High Friction". 
        Residents in these areas are likely engaged in manual labor or have poor historical fingerprint quality.</p>
        <p><b>Recommendation:</b> Shift from Fingerprint-first to <b>Iris-first authentication</b> hardware in the following zones immediately.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Dataframe with gradients
    st.subheader("Priority Deployment List")
    
    display_cols = ['state', 'district', 'Total_Biometric_Updates', 'Total_Demographic_Updates', 'Total_Enrolments', 'Friction_Index']
    
    # Custom styling for dataframe to look good in dark mode
    # Streamlit dataframe doesn't support full CSS injection easily, but we can style the content
    st.dataframe(
        critical_districts[display_cols].style.background_gradient(subset=['Friction_Index'], cmap='Reds')
                                      .format({'Friction_Index': "{:.2f}", 'Total_Biometric_Updates': "{:,.0f}", 'Total_Enrolments': "{:,.0f}"}),
        use_container_width=True
    )

    # --- Download Button ---
    csv = critical_districts.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Priority List (CSV)",
        data=csv,
        file_name='high_friction_districts_priority.csv',
        mime='text/csv',
    )