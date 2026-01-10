import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="UIDAI Pulse 2026: The Identity Anxiety Spectrum",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a Hackathon-Winning Aesthetic
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #1E232F;
        border: 1px solid #2C3342;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #00D4FF;
        font-size: 28px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FAFAFA;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00D4FF, #0055FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #131720;
        border-right: 1px solid #2C3342;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E232F;
        border-radius: 4px;
        color: white;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00D4FF;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & PROCESSING ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_process_data():
    try:
        # Load Files
        # Note: Using the exact filenames provided
        bio_files = ['api_data_aadhar_biometric_0_500000.csv', 'api_data_aadhar_biometric_500000_1000000.csv']
        demo_files = ['api_data_aadhar_demographic_1000000_1500000.csv', 'api_data_aadhar_demographic_1500000_2000000.csv']
        enrol_files = ['api_data_aadhar_enrolment_0_500000.csv', 'api_data_aadhar_enrolment_500000_1000000.csv']

        df_bio = pd.concat((pd.read_csv(f) for f in bio_files), ignore_index=True)
        df_demo = pd.concat((pd.read_csv(f) for f in demo_files), ignore_index=True)
        df_enrol = pd.concat((pd.read_csv(f) for f in enrol_files), ignore_index=True)

        # Basic Cleanup
        for df in [df_bio, df_demo, df_enrol]:
            df.columns = [c.strip() for c in df.columns] # Clean whitespace
            # Standardize State/District names (Simple capitalization for consistency)
            df['state'] = df['state'].str.title().str.strip()
            df['district'] = df['district'].str.title().str.strip()

        # Aggregation by State & District
        # Biometric Focus: Adult Updates (likely aging/mandatory) -> bio_age_17_
        bio_grp = df_bio.groupby(['state', 'district'])[['bio_age_17_', 'bio_age_5_17']].sum().reset_index()
        
        # Demographic Focus: Adult Updates (Corrections/KYC) -> demo_age_17_
        demo_grp = df_demo.groupby(['state', 'district'])[['demo_age_17_', 'demo_age_5_17']].sum().reset_index()
        
        # Enrolment Focus: New Adults (Inclusion) -> age_18_greater
        enrol_grp = df_enrol.groupby(['state', 'district'])[['age_18_greater']].sum().reset_index()

        # Master Merge
        master = pd.merge(bio_grp, demo_grp, on=['state', 'district'], how='outer').fillna(0)
        master = pd.merge(master, enrol_grp, on=['state', 'district'], how='outer').fillna(0)

        # -------------------------------------------------------------------------
        # THE CORE ALGORITHM: Demographic-Biometric Divergence Index (DBDI)
        # -------------------------------------------------------------------------
        # Logic: 
        # High Demo / Low Bio = Identity Anxiety (Active Correction)
        # Low Demo / High Bio = Digital Dormancy (Passive Compliance)
        
        # Avoid division by zero
        master['bio_age_17_'] = master['bio_age_17_'].replace(0, 1) 
        
        master['DBDI'] = master['demo_age_17_'] / master['bio_age_17_']
        master['Total_Activity'] = master['demo_age_17_'] + master['bio_age_17_']

        # Classification Logic
        def classify_district(row):
            ratio = row['DBDI']
            if ratio > 2.5:
                return "Cluster A: Hyper-Correction (Identity Anxiety)"
            elif ratio < 0.2:
                return "Cluster B: Digital Dormancy (Passive Compliance)"
            else:
                return "Cluster C: Balanced Economy"

        master['Cluster'] = master.apply(classify_district, axis=1)
        
        return master

    except Exception as e:
        st.error(f"Data Loading Error: {str(e)}")
        return pd.DataFrame()

df = load_and_process_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Analysis Controls")
    st.markdown("---")
    
    # State Filter
    all_states = sorted(df['state'].unique().tolist())
    selected_states = st.multiselect("Select States (Default: All)", all_states)
    
    # Cluster Filter
    all_clusters = sorted(df['Cluster'].unique().tolist())
    selected_clusters = st.multiselect("Select Cluster Type", all_clusters, default=all_clusters)
    
    # Volumetric Filter (To remove noise)
    min_volume = st.slider("Min Total Activity (to exclude outliers)", 0, 10000, 1000)
    
    st.markdown("---")
    st.info("""
    **Understanding the DBDI:**
    
    **> 2.5 (Red):** High Correction. People are fixing names/addresses aggressively. Signals strict KYC or documentation fears.
    
    **< 0.2 (Blue):** High Compliance. People only update biometrics when forced. Signals low digital usage.
    """)
    
    st.caption("UIDAI Data Hackathon 2026 | Prototype v1.0")

# Apply Filters
df_filtered = df[df['Total_Activity'] >= min_volume]
if selected_states:
    df_filtered = df_filtered[df_filtered['state'].isin(selected_states)]
if selected_clusters:
    df_filtered = df_filtered[df_filtered['Cluster'].isin(selected_clusters)]

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD UI
# -----------------------------------------------------------------------------

# Header
st.title("🇮🇳 The 'Identity Anxiety' Spectrum")
st.markdown("""
<div style='font-size: 1.2em; color: #B0B3B8; margin-bottom: 20px;'>
    Decoding the 65x Gap in Digital Behavior Between Border Districts and Tribal Belts.
    <br><b>The Insight:</b> While Metros run a "Maintenance Economy", Border districts run a "Correction Economy".
</div>
""", unsafe_allow_html=True)

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Biometric Updates", f"{df_filtered['bio_age_17_'].sum():,.0f}", "Passive Push")
with col2:
    st.metric("Total Demographic Updates", f"{df_filtered['demo_age_17_'].sum():,.0f}", "Active Pull")
with col3:
    avg_dbdi = df_filtered['demo_age_17_'].sum() / df_filtered['bio_age_17_'].sum()
    st.metric("Avg. Divergence Index (DBDI)", f"{avg_dbdi:.2f}", "National Baseline")
with col4:
    anxiety_districts = len(df_filtered[df_filtered['Cluster'].str.contains("Hyper-Correction")])
    st.metric("High Anxiety Districts", anxiety_districts, "Need Intervention")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 The Divergence Matrix", "🗺️ Geographic Hotspots", "🧠 Strategic Intelligence"])

# --- TAB 1: THE SCATTER PLOT (THE WINNING VISUAL) ---
with tab1:
    st.markdown("### The Two Indias: Active Correction vs. Passive Compliance")
    
    # Advanced Scatter Plot using Plotly Express
    fig_scatter = px.scatter(
        df_filtered,
        x="bio_age_17_",
        y="demo_age_17_",
        color="Cluster",
        size="Total_Activity",
        hover_name="district",
        hover_data=["state", "DBDI", "age_18_greater"],
        log_x=True, 
        log_y=True,
        color_discrete_map={
            "Cluster A: Hyper-Correction (Identity Anxiety)": "#FF4B4B",
            "Cluster B: Digital Dormancy (Passive Compliance)": "#00D4FF",
            "Cluster C: Balanced Economy": "#FFAA00"
        },
        labels={
            "bio_age_17_": "Biometric Updates (Log Scale)",
            "demo_age_17_": "Demographic Updates (Log Scale)"
        },
        template="plotly_dark",
        height=600
    )
    
    # Add diagonal line (1:1 Ratio)
    fig_scatter.add_shape(
        type="line", line=dict(dash="dash", color="white", width=1),
        x0=100, y0=100, x1=1000000, y1=1000000
    )
    
    fig_scatter.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("""
    > **How to Read this Chart:**
    > * **Above the Diagonal (Red):** Districts where demographic corrections outpace biometric updates. These are "Hotspots" of identity stress.
    > * **Below the Diagonal (Blue):** Districts where biometric updates dominate. These are "Coldspots" of digital exclusion.
    > * **On the Diagonal (Yellow):** Healthy, balanced digital ecosystems.
    """)

# --- TAB 2: GEOGRAPHIC HOTSPOTS ---
with tab2:
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("### Top 'Hyper-Correction' Zones (High Anxiety)")
        top_anxiety = df_filtered[df_filtered['Cluster'].str.contains("Hyper-Correction")].sort_values(by='DBDI', ascending=False).head(10)
        
        fig_bar_anx = px.bar(
            top_anxiety,
            x='DBDI',
            y='district',
            orientation='h',
            color='DBDI',
            color_continuous_scale='Reds',
            text='state',
            title="Highest Ratio of Corrections to Updates",
            template="plotly_dark"
        )
        fig_bar_anx.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar_anx, use_container_width=True)
        
    with row1_col2:
        st.markdown("### Top 'Digital Dormancy' Zones (Low Usage)")
        top_dormant = df_filtered[df_filtered['Cluster'].str.contains("Dormancy")].sort_values(by='DBDI', ascending=True).head(10)
        
        fig_bar_dor = px.bar(
            top_dormant,
            x='DBDI',
            y='district',
            orientation='h',
            color='DBDI',
            color_continuous_scale='Blues_r',
            text='state',
            title="Lowest Ratio of Corrections to Updates",
            template="plotly_dark"
        )
        fig_bar_dor.update_layout(yaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_bar_dor, use_container_width=True)

    st.markdown("### State-wise Breakdown")
    state_pivot = df_filtered.groupby('state')[['bio_age_17_', 'demo_age_17_']].sum()
    state_pivot['State_DBDI'] = state_pivot['demo_age_17_'] / state_pivot['bio_age_17_']
    state_pivot = state_pivot.sort_values(by='State_DBDI', ascending=False)
    
    fig_state = px.bar(
        state_pivot.reset_index(),
        x='state',
        y='State_DBDI',
        color='State_DBDI',
        color_continuous_scale='RdYlBu_r',
        title="Average Divergence Index by State",
        template="plotly_dark"
    )
    # Add threshold line
    fig_state.add_hline(y=1.0, line_dash="dot", line_color="white", annotation_text="Balanced (1.0)")
    st.plotly_chart(fig_state, use_container_width=True)

# --- TAB 3: STRATEGIC INTELLIGENCE ---
with tab3:
    st.markdown("## 🧠 Actionable Recommendations")
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.warning("### 🚩 For Cluster A (Hyper-Correction)")
        st.markdown("""
        **The Problem:** High volume of demographic changes suggests high rejection rates and citizen panic regarding documentation (e.g., Border districts, Welfare verification).
        
        **The Solution:**
        1. **Deploy Mobile Correction Camps:** Move resources from Enrolment to Correction in these districts.
        2. **Pre-verification Drives:** Partner with local banks/schools to pre-verify documents before the user reaches the center.
        3. **Focus:** Reduce 'Retry' loops.
        """)
        
    with col_rec2:
        st.info("### 🧊 For Cluster B (Digital Dormancy)")
        st.markdown("""
        **The Problem:** Users maintain biometrics (forced) but don't update demographics, implying they aren't using the ID for dynamic services like Banking or Jobs.
        
        **The Solution:**
        1. **Utility Campaigns:** Link Aadhaar to local benefits to drive 'active' usage.
        2. **Banking Integration:** These districts likely have high 'unbanked' populations. Cross-reference with Financial Inclusion data.
        3. **Focus:** Increase 'Utility'.
        """)

    st.markdown("---")
    st.markdown("### 📥 Export Processed Data")
    
    st.dataframe(
        df_filtered[['state', 'district', 'bio_age_17_', 'demo_age_17_', 'age_18_greater', 'DBDI', 'Cluster']].sort_values(by='DBDI', ascending=False),
        use_container_width=True
    )
    
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_filtered)

    st.download_button(
        label="Download Analysis CSV",
        data=csv,
        file_name='uidai_hackathon_analysis.csv',
        mime='text/csv',
    )