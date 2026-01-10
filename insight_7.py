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
    initial_sidebar_state="collapsed" 
)

# Custom CSS for the Navy Blue Gradient & "Hackathon" Aesthetic
st.markdown("""
    <style>
    /* Main Background - Navy Blue Gradient */
    .stApp {
        background: #000428;  /* fallback for old browsers */
        background: -webkit-linear-gradient(to right, #004e92, #000428);  /* Chrome 10-25, Safari 5.1-6 */
        background: linear-gradient(to right, #004e92, #000428); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
        color: #E0E0E0;
    }
    
    /* Metrics Cards Styling */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #00D4FF;
    }
    div[data-testid="stMetricValue"] {
        color: #00D4FF; /* Cyan for numbers */
        font-size: 28px;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        color: #B0B3B8;
    }
    
    /* Headers Styling */
    h1, h2, h3, h4 {
        color: #FAFAFA;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00D4FF, #0055FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Custom Navigation Bar Container (Expander) */
    .streamlit-expanderHeader {
        background-color: rgba(0, 20, 40, 0.8);
        border: 1px solid #00D4FF;
        color: #00D4FF !important;
        font-weight: bold;
        border-radius: 5px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #00091a;
        border-right: 1px solid #1a2639;
    }
    
    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & PROCESSING ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_process_data():
    """
    Loads biometric, demographic, and enrolment data from CSVs.
    Performs merging, cleaning, and calculates the Divergence Index.
    """
    try:
        # Load Files (Now including all segments provided for complete analysis)
        bio_files = [
            'api_data_aadhar_biometric_0_500000.csv', 
            'api_data_aadhar_biometric_500000_1000000.csv',
            'api_data_aadhar_biometric_1000000_1500000.csv',
            'api_data_aadhar_biometric_1500000_1861108.csv'
        ]
        
        demo_files = [
            'api_data_aadhar_demographic_0_500000.csv',
            'api_data_aadhar_demographic_500000_1000000.csv',
            'api_data_aadhar_demographic_1000000_1500000.csv', 
            'api_data_aadhar_demographic_1500000_2000000.csv',
            'api_data_aadhar_demographic_2000000_2071700.csv'
        ]
        
        enrol_files = [
            'api_data_aadhar_enrolment_0_500000.csv', 
            'api_data_aadhar_enrolment_500000_1000000.csv',
            'api_data_aadhar_enrolment_1000000_1006029.csv'
        ]

        # Generator expressions for efficient concatenation
        # We use a try-except inside the generator to handle potential missing files gracefully if running locally without all chunks
        def safe_read(f):
            try:
                return pd.read_csv(f)
            except FileNotFoundError:
                return pd.DataFrame()

        df_bio = pd.concat((safe_read(f) for f in bio_files), ignore_index=True)
        df_demo = pd.concat((safe_read(f) for f in demo_files), ignore_index=True)
        df_enrol = pd.concat((safe_read(f) for f in enrol_files), ignore_index=True)

        # Basic Cleanup & Standardization
        for df in [df_bio, df_demo, df_enrol]:
            if not df.empty:
                df.columns = [c.strip() for c in df.columns] # Clean whitespace
                # Standardize State/District names (Title case for consistency)
                if 'state' in df.columns:
                    df['state'] = df['state'].str.title().str.strip()
                if 'district' in df.columns:
                    df['district'] = df['district'].str.title().str.strip()

        # Aggregation by State & District
        # Biometric Focus: Adult Updates (likely aging/mandatory) -> bio_age_17_
        if not df_bio.empty:
            bio_grp = df_bio.groupby(['state', 'district'])[['bio_age_17_', 'bio_age_5_17']].sum().reset_index()
        else:
            bio_grp = pd.DataFrame(columns=['state', 'district', 'bio_age_17_', 'bio_age_5_17'])
        
        # Demographic Focus: Adult Updates (Corrections/KYC) -> demo_age_17_
        if not df_demo.empty:
            demo_grp = df_demo.groupby(['state', 'district'])[['demo_age_17_', 'demo_age_5_17']].sum().reset_index()
        else:
            demo_grp = pd.DataFrame(columns=['state', 'district', 'demo_age_17_', 'demo_age_5_17'])
        
        # Enrolment Focus: New Adults (Inclusion) -> age_18_greater
        if not df_enrol.empty:
            enrol_grp = df_enrol.groupby(['state', 'district'])[['age_18_greater']].sum().reset_index()
        else:
             enrol_grp = pd.DataFrame(columns=['state', 'district', 'age_18_greater'])

        # Master Merge: Outer join to ensure no district is left behind
        master = pd.merge(bio_grp, demo_grp, on=['state', 'district'], how='outer').fillna(0)
        master = pd.merge(master, enrol_grp, on=['state', 'district'], how='outer').fillna(0)

        # -------------------------------------------------------------------------
        # THE CORE ALGORITHM: Demographic-Biometric Divergence Index (DBDI)
        # -------------------------------------------------------------------------
        # Logic: 
        # High Demo / Low Bio = Identity Anxiety (Active Correction)
        # Low Demo / High Bio = Digital Dormancy (Passive Compliance)
        
        # Avoid division by zero by replacing 0 with 1 in denominator
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
        # Return empty DF structure to prevent app crash
        return pd.DataFrame(columns=['state', 'district', 'bio_age_17_', 'demo_age_17_', 'age_18_greater', 'DBDI', 'Total_Activity', 'Cluster'])

# Load the data
df = load_and_process_data()

# -----------------------------------------------------------------------------
# 3. NAVIGATION BAR & CONTROLS (Moved to Top as requested)
# -----------------------------------------------------------------------------

# We use an Expander to simulate a collapsible Top Navigation Bar/Control Panel
with st.expander("🎛️ NAVIGATION & ANALYSIS CONTROLS (Open to Filter Data)", expanded=True):
    st.markdown("##### 🛠️ Configure Your Data View")
    
    # Using columns to create a horizontal layout for controls
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        # State Filter
        all_states = sorted(df['state'].unique().tolist()) if not df.empty else []
        selected_states = st.multiselect("📍 Filter by States", all_states, placeholder="All India")
    
    with nav_col2:
        # Cluster Filter
        all_clusters = sorted(df['Cluster'].unique().tolist()) if not df.empty else []
        selected_clusters = st.multiselect("📊 Filter by Cluster Type", all_clusters, default=all_clusters)
        
    with nav_col3:
        # Volumetric Filter (To remove noise)
        min_volume = st.slider("📉 Min. Transaction Volume (Outlier Removal)", 0, 10000, 1000)

    # Explanation of Filters in the navbar area
    st.caption("Use these controls to slice the dataset. The 'Identity Anxiety' spectrum analyzes the ratio between demographic corrections (Anxiety) and biometric updates (Compliance).")

# Apply Filters Logic
df_filtered = df.copy()
# Filter by Volume
df_filtered = df_filtered[df_filtered['Total_Activity'] >= min_volume]
# Filter by State
if selected_states:
    df_filtered = df_filtered[df_filtered['state'].isin(selected_states)]
# Filter by Cluster
if selected_clusters:
    df_filtered = df_filtered[df_filtered['Cluster'].isin(selected_clusters)]

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD UI (Sidebar Removed)
# -----------------------------------------------------------------------------

# Header Section
st.title("🇮🇳 The 'Identity Anxiety' Spectrum")
st.markdown("""
<div style='font-size: 1.3em; color: #B0B3B8; margin-bottom: 20px; border-left: 5px solid #00D4FF; padding-left: 15px;'>
    Decoding the 65x Gap in Digital Behavior Between Border Districts and Tribal Belts.
    <br><span style='color: #E0E0E0; font-size: 0.9em;'><b>The Insight:</b> While Metros run a "Maintenance Economy", Border districts run a "Correction Economy".</span>
</div>
""", unsafe_allow_html=True)

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Biometric Updates", f"{df_filtered['bio_age_17_'].sum():,.0f}", "Passive Push")
with col2:
    st.metric("Total Demographic Updates", f"{df_filtered['demo_age_17_'].sum():,.0f}", "Active Pull")
with col3:
    # Safe division
    total_demo = df_filtered['demo_age_17_'].sum()
    total_bio = df_filtered['bio_age_17_'].sum()
    avg_dbdi = total_demo / total_bio if total_bio > 0 else 0
    st.metric("Avg. Divergence Index (DBDI)", f"{avg_dbdi:.2f}", "National Baseline")
with col4:
    anxiety_districts = len(df_filtered[df_filtered['Cluster'].str.contains("Hyper-Correction")])
    st.metric("High Anxiety Districts", anxiety_districts, "Need Intervention")

# -----------------------------------------------------------------------------
# 5. SINGLE PAGE LAYOUT (TABS REMOVED)
# -----------------------------------------------------------------------------

st.markdown("---")

# --- SECTION 1: THE SCATTER PLOT ---
st.header("📊 The Divergence Matrix")
st.markdown("### The Two Indias: Active Correction vs. Passive Compliance")

if not df_filtered.empty:
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("""
    > **How to Read this Chart:**
    > * **Above the Diagonal (Red):** Districts where demographic corrections outpace biometric updates. These are "Hotspots" of identity stress.
    > * **Below the Diagonal (Blue):** Districts where biometric updates dominate. These are "Coldspots" of digital exclusion.
    > * **On the Diagonal (Yellow):** Healthy, balanced digital ecosystems.
    """)
else:
    st.warning("No data available for the current selection.")

st.markdown("---")

# --- SECTION 2: GEOGRAPHIC HOTSPOTS ---
st.header("🗺️ Geographic Hotspots")

if not df_filtered.empty:
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
        fig_bar_anx.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
        fig_bar_dor.update_layout(yaxis={'categoryorder':'total descending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
    fig_state.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_state, use_container_width=True)

st.markdown("---")

# --- SECTION 3: STATISTICAL DEEP DIVE ---
st.header("📈 Statistical Deep Dive")
st.markdown("Here we deconstruct the data using Univariate, Bivariate, and Trivariate methods to find hidden correlations.")

# 1. UNIVARIATE ANALYSIS
st.markdown("### 1. Univariate Analysis: Understanding the Distributions")
st.info("**What is it?** Univariate analysis explores a single variable at a time. Here we look at the spread of the **DBDI (Divergence Index)** to see if Identity Anxiety is a rare anomaly or a common trend.")

col_uni1, col_uni2 = st.columns(2)

with col_uni1:
    # Histogram of DBDI
    fig_hist = px.histogram(
        df_filtered, 
        x="DBDI", 
        nbins=50, 
        title="Distribution of Divergence Index (DBDI)",
        color_discrete_sequence=['#00D4FF'],
        template="plotly_dark"
    )
    fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist, use_container_width=True)

with col_uni2:
    # Box Plot of Total Activity
    fig_box_act = px.box(
        df_filtered, 
        y="Total_Activity", 
        title="Spread of Total Transaction Volume",
        color_discrete_sequence=['#FFAA00'],
        template="plotly_dark"
    )
    fig_box_act.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_box_act, use_container_width=True)

# 2. BIVARIATE ANALYSIS
st.markdown("### 2. Bivariate Analysis: finding Relationships")
st.info("**What is it?** Bivariate analysis compares two variables. We check if higher *New Enrolments* lead to higher *Updates*, and how the *DBDI* varies across *States*.")

col_bi1, col_bi2 = st.columns(2)

with col_bi1:
    # Scatter: Enrolment vs Total Activity
    fig_bi_scat = px.scatter(
        df_filtered,
        x="age_18_greater",
        y="Total_Activity",
        color="Cluster",
        title="Correlation: New Enrolments vs. Update Volume",
        labels={"age_18_greater": "New Adult Enrolments", "Total_Activity": "Total Updates"},
        template="plotly_dark",
        log_x=True, log_y=True
    )
    fig_bi_scat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bi_scat, use_container_width=True)

with col_bi2:
    # Boxplot: DBDI by Cluster
    fig_bi_box = px.box(
        df_filtered,
        x="Cluster",
        y="DBDI",
        color="Cluster",
        title="Divergence Index Intensity by Cluster",
        template="plotly_dark"
    )
    fig_bi_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig_bi_box, use_container_width=True)

# 3. TRIVARIATE ANALYSIS
st.markdown("### 3. Trivariate Analysis: Multi-Dimensional View")
st.info("**What is it?** Trivariate analysis looks at three variables simultaneously. Below, we plot **Biometric Updates (X)** vs **Demographic Updates (Y)** vs **New Enrolments (Z)** to see the full 3D ecosystem.")

if not df_filtered.empty:
    # 3D Scatter Plot
    fig_3d = px.scatter_3d(
        df_filtered,
        x='bio_age_17_',
        y='demo_age_17_',
        z='age_18_greater',
        color='Cluster',
        size='Total_Activity', 
        size_max=30,
        opacity=0.7,
        title="3D Interaction: Bio vs. Demo vs. Enrolment",
        labels={
            'bio_age_17_': 'Bio Updates', 
            'demo_age_17_': 'Demo Updates',
            'age_18_greater': 'New Enrolments'
        },
        template="plotly_dark",
        color_discrete_map={
            "Cluster A: Hyper-Correction (Identity Anxiety)": "#FF4B4B",
            "Cluster B: Digital Dormancy (Passive Compliance)": "#00D4FF",
            "Cluster C: Balanced Economy": "#FFAA00"
        }
    )
    fig_3d.update_layout(height=700, paper_bgcolor='rgba(0,0,0,0)', scene=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# --- SECTION 4: STRATEGIC INTELLIGENCE ---
st.header("🧠 Strategic Intelligence")
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

st.markdown("---")

# --- SECTION 5: MODULE DOCUMENTATION ---
st.header("📘 Module Documentation")
st.markdown("## 📘 Detailed Module Explanation")
st.markdown("### 1. The Core Philosophy: 'Identity Anxiety'")
st.write("This module operates on a sociological hypothesis: **The way citizens interact with their Digital ID reveals their socio-economic stability.**")

st.markdown("### 2. The Algorithm: DBDI")
st.latex(r'''
    DBDI = \frac{\text{Demographic Updates (Corrections)}}{\text{Biometric Updates (Mandatory)}}
''')
st.write("""
* **Numerator (Demographic Updates):** Changes to Name, Address, DOB, Gender. These are *voluntary* and *corrective*. A high number implies the ID was rejected somewhere (Bank, Ration, Job), forcing the user to fix it. This represents **"Friction"**.
* **Denominator (Biometric Updates):** Mandatory updates at age 5 and 15, or every 10 years. These are *compliance-driven*. A high number without demographic changes implies the user is just following rules but not actively using the ID for services that require accurate demographic data. This represents **"Compliance"**.
""")

st.markdown("### 3. Cluster Definitions")
col_doc1, col_doc2, col_doc3 = st.columns(3)
with col_doc1:
    st.error("#### Cluster A: Hyper-Correction")
    st.write("**DBDI > 2.5**")
    st.write("Districts where people are frantically correcting their IDs. Often seen in border areas (citizenship anxiety) or migration hubs (address changes).")
with col_doc2:
    st.info("#### Cluster B: Digital Dormancy")
    st.write("**DBDI < 0.2**")
    st.write("Districts where people update biometrics only when forced by the system, but rarely fix demographic errors. Suggests the ID is sitting in a drawer, unused for financial or employment access.")
with col_doc3:
    st.warning("#### Cluster C: Balanced Economy")
    st.write("**0.2 <= DBDI <= 2.5**")
    st.write("A healthy digital state where updates happen naturally alongside mandatory biometric refreshes.")

st.markdown("### 4. Data Dictionary & Sources")
st.write("This module processes synthetic API data mirroring the structure of UIDAI datasets.")
st.dataframe(pd.DataFrame({
    "Variable": ["bio_age_17_", "demo_age_17_", "age_18_greater", "Total_Activity"],
    "Description": ["Biometric updates for adults (Compliance)", "Demographic corrections for adults (Anxiety)", "New adult enrolments (Inclusion)", "Sum of Bio + Demo updates (Volume)"],
    "Source File Category": ["Biometric API", "Demographic API", "Enrolment API", "Calculated Field"]
}), use_container_width=True)

st.markdown("### 5. Why this matters?")
st.info("""
Traditional metrics track **Volume** (How many updates?). 
This module tracks **Intent** (Why are they updating?).

By separating "Compliance" from "Correction", we can allocate resources better:
* **High Anxiety Areas** need more operators and help desks.
* **High Dormancy Areas** need more utility and banking integration.
""")