import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Aadhar Forensic Analytics: Ghost Village Detector",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# --- Custom Styling (Navy Blue Gradient & Navigation Bar) ---
st.markdown("""
<style>
    /* Main App Background - Navy Blue Gradient */
    .stApp {
        background: #000428;  /* fallback for old browsers */
        background: -webkit-linear-gradient(to right, #004e92, #000428);  /* Chrome 10-25, Safari 5.1-6 */
        background: linear-gradient(to right, #004e92, #000428); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
        color: #e0f7fa;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 { 
        color: #00d2ff !important; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Text overrides */
    p, label, .stMarkdown {
        color: #b3e5fc !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #00d2ff; 
        color: #000428; 
        border-radius: 8px; 
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #ffffff;
        color: #000428;
        transform: scale(1.02);
    }

    /* Metric Cards */
    div.stMetric {
        background-color: rgba(0, 0, 0, 0.4); 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #00d2ff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div.stMetric > div > div > div {
        color: #e0f7fa !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05);
        color: #00d2ff !important;
        border-radius: 5px;
    }

    /* Risk Classes for Text */
    .risk-high { color: #ff5252; font-weight: bold; }
    .risk-med { color: #ffd740; font-weight: bold; }
    .risk-low { color: #69f0ae; font-weight: bold; }

    /* Navigation/Filter Bar Container Styling (simulated via CSS on specific elements if needed, but standard columns work well) */
    
</style>
""", unsafe_allow_html=True)

# --- Data Loading & Preprocessing ---
@st.cache_data
def load_and_process_data():
    try:
        # 1. Load Enrolment Data (New Entries)
        enrol_files = [
            'api_data_aadhar_enrolment_0_500000.csv',
            'api_data_aadhar_enrolment_500000_1000000.csv'
        ]
        df_enrol = pd.concat([pd.read_csv(f) for f in enrol_files])
        
        # Calculate Total Enrolments per Pincode
        df_enrol['total_enrolments'] = (
            df_enrol['age_0_5'] + df_enrol['age_5_17'] + df_enrol['age_18_greater']
        )
        # Group by Pincode/State/District to get unique locations
        enrol_grouped = df_enrol.groupby(['state', 'district', 'pincode'])['total_enrolments'].sum().reset_index()

        # 2. Load Biometric & Demographic Data (Updates)
        # We treat both Biometric and Demographic logs as "Proof of Life/Activity"
        bio_files = [
            'api_data_aadhar_biometric_0_500000.csv',
            'api_data_aadhar_biometric_500000_1000000.csv'
        ]
        demo_files = [
            'api_data_aadhar_demographic_1000000_1500000.csv',
            'api_data_aadhar_demographic_1500000_2000000.csv'
        ]
        
        df_bio = pd.concat([pd.read_csv(f) for f in bio_files])
        df_demo = pd.concat([pd.read_csv(f) for f in demo_files])

        # Sum updates
        df_bio['total_bio_updates'] = df_bio['bio_age_5_17'] + df_bio['bio_age_17_']
        df_demo['total_demo_updates'] = df_demo['demo_age_5_17'] + df_demo['demo_age_17_']

        # Group updates by Pincode
        bio_grouped = df_bio.groupby('pincode')['total_bio_updates'].sum().reset_index()
        demo_grouped = df_demo.groupby('pincode')['total_demo_updates'].sum().reset_index()

        # 3. Merge Datasets to create the Master Forensic Table
        master_df = pd.merge(enrol_grouped, bio_grouped, on='pincode', how='left').fillna(0)
        master_df = pd.merge(master_df, demo_grouped, on='pincode', how='left').fillna(0)

        # Total Updates Calculation
        master_df['total_updates'] = master_df['total_bio_updates'] + master_df['total_demo_updates']

        return master_df

    except FileNotFoundError as e:
        st.error(f"Critical Error: Missing Dataset Files. Please ensure all CSVs are in the directory. Details: {e}")
        return pd.DataFrame()

# --- Advanced Analytics Engine ---
def perform_cluster_analysis(df, n_clusters=3):
    """
    Uses K-Means Clustering to group pincodes based on Enrolment vs Update behavior.
    Feature scaling is applied first.
    """
    if df.empty:
        return df

    # Select features for clustering
    features = df[['total_enrolments', 'total_updates']]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Label Clusters dynamically (High Enrolment/Low Update = High Risk)
    # We calculate the mean 'Update Ratio' for each cluster to identify which is which
    df['update_ratio'] = df['total_updates'] / (df['total_enrolments'] + 1) # +1 to avoid div/0
    
    cluster_stats = df.groupby('cluster')['update_ratio'].mean().sort_values()
    
    # Map cluster IDs to risk labels based on update ratio (Low ratio = High Risk)
    risk_mapping = {
        cluster_stats.index[0]: 'High Risk (Ghost Village)', 
        cluster_stats.index[1]: 'Medium Risk (Monitor)', 
        cluster_stats.index[2]: 'Low Risk (Normal Activity)'
    }
    
    df['Risk_Profile'] = df['cluster'].map(risk_mapping)
    return df

# --- UI Layout ---

def main():
    # Load Data First
    df = load_and_process_data()
    
    if df.empty:
        return

    # --- Header ---
    st.title("🕵️‍♂️ Project: Ghost Village Detector")
    
    # --- TOP NAVIGATION BAR (Filters moved from Sidebar) ---
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="margin-top:0; color: #00d2ff;">🔍 Forensic Navigation & Filters</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Create 3 columns for the Navigation Bar
    nav_col1, nav_col2, nav_col3 = st.columns(3)

    # 1. State Selector
    with nav_col1:
        all_states = sorted(df['state'].unique())
        selected_state = st.selectbox("📍 Select State", ["All States"] + all_states)
    
    # 2. District Selector (Dependent)
    with nav_col2:
        if selected_state != "All States":
            df_filtered_pre = df[df['state'] == selected_state]
            all_districts = sorted(df_filtered_pre['district'].unique())
            selected_district = st.multiselect("🏙️ Select District", all_districts, default=all_districts)
        else:
            df_filtered_pre = df.copy()
            selected_district = []
            st.info("Select a State to filter by District")

    # 3. Sensitivity Slider
    with nav_col3:
        min_enrolments = st.slider("⚖️ Min Enrolment Threshold", 0, 500, 50, help="Filter out tiny hamlets.")

    # --- Apply Filters Logic ---
    if selected_state != "All States":
        # If districts are selected, filter by them
        if selected_district:
            df_filtered = df_filtered_pre[df_filtered_pre['district'].isin(selected_district)]
        else:
            # If state selected but no district (or user deselected all), show state data
            df_filtered = df_filtered_pre
    else:
        df_filtered = df.copy()

    # Apply Sensitivity Filter
    df_filtered = df_filtered[df_filtered['total_enrolments'] >= min_enrolments]

    # --- Run Analytics ---
    # Run ML Clustering on Filtered Data
    if len(df_filtered) > 5:
        df_analyzed = perform_cluster_analysis(df_filtered)
    else:
        df_analyzed = df_filtered
        df_analyzed['Risk_Profile'] = "Insufficient Data for ML"

    # --- Dashboard Context ---
    st.markdown("""
    **Objective:** Identify pincodes with high `Enrolment` activity but disproportionately low `Biometric/Demographic Updates`. 
    *Hypothesis: Real residents eventually need updates. Zero-update zones with high enrolments may indicate fake entity generation farms.*
    """)

    # --- Educational Module: Concept ---
    with st.expander("ℹ️ About the 'Ghost Village' Methodology"):
        st.markdown("""
        ### 🧠 Forensic Concept: Lifecycle Anomaly Detection
        In a healthy population, high Aadhar enrolment leads to a predictable volume of subsequent updates (changes in address, mobile number, biometrics).
        
        **The 'Ghost Village' Anomaly:**
        When a specific pincode shows massive Enrolment numbers but **near-zero** Biometric or Demographic updates, it suggests the entities might be:
        1.  **Non-existent/Fake:** Created solely for benefit diversion.
        2.  **Dormant:** Created once and never used again.
        
        **Machine Learning Approach:**
        We use **K-Means Clustering** (Unsupervised Learning) to automatically detect these natural groupings without human bias. It mathematically separates 'Normal' behavior from 'Suspicious' behavior based on the ratio of inputs (Enrolments) to lifecycle events (Updates).
        """)

    # --- KPI Row ---
    st.subheader("📊 Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    
    total_pincodes = len(df_analyzed)
    high_risk_count = len(df_analyzed[df_analyzed['Risk_Profile'] == 'High Risk (Ghost Village)'])
    total_enrol = df_analyzed['total_enrolments'].sum()
    suspicious_enrol = df_analyzed[df_analyzed['Risk_Profile'] == 'High Risk (Ghost Village)']['total_enrolments'].sum()

    col1.metric("Total Pincodes Scanned", f"{total_pincodes:,}")
    col2.metric("🚩 High Risk Pincodes", f"{high_risk_count}", delta_color="inverse")
    col3.metric("Total Enrolments", f"{total_enrol:,}")
    col4.metric("⚠️ Enrolments in Risk Zones", f"{suspicious_enrol:,}", delta_color="inverse")

    st.markdown("---")

    # --- Main Visualization: Cluster Analysis ---
    st.subheader("1. Anomaly Cluster Detection")
    st.markdown("Visualizing Pincodes by **Activity Volume** vs. **Updates**. The 'Ghost' zones are pinned to the bottom right (High Enrolment, Zero Updates).")
    
    with st.expander("ℹ️ Visualization Guide: Scatter Plot (Bivariate Analysis)"):
        st.markdown("""
        **Type of Analysis:** **Bivariate Analysis** (with Multivariate Clustering).
        * **X-Axis (Enrolments):** Represents the volume of new entries.
        * **Y-Axis (Updates):** Represents the volume of maintenance activity.
        * **Color (Cluster):** The 3rd dimension derived from ML, indicating Risk Profile.
        
        **How to read this:**
        * **Top Right (Green/Low Risk):** High Enrolment AND High Updates. This is a busy, real city.
        * **Bottom Right (Red/High Risk):** High Enrolment but Zero/Low Updates. This is the **Ghost Village Zone**.
        * **Bottom Left:** Low activity overall.
        """)

    fig_scatter = px.scatter(
        df_analyzed,
        x="total_enrolments",
        y="total_updates",
        color="Risk_Profile",
        size="total_enrolments",
        hover_data=["pincode", "district", "state"],
        color_discrete_map={
            "High Risk (Ghost Village)": "#ff5252",
            "Medium Risk (Monitor)": "#ffd740",
            "Low Risk (Normal Activity)": "#69f0ae"
        },
        title="Forensic Scatter: Enrolment Volume vs. Life-Cycle Updates",
        height=600
    )
    # Update chart to match Navy Theme
    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0f7fa",
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        xaxis_title="Total Enrolments (New Entries)",
        yaxis_title="Total Updates (Proof of Life)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Drill Down Section ---
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("2. Risk Distribution by District")
        
        with st.expander("ℹ️ Visualization Guide: Bar Chart (Univariate Analysis)"):
            st.markdown("""
            **Type of Analysis:** **Univariate Analysis** (Frequency Distribution).
            * **Variable:** District Name (Categorical).
            * **Metric:** Count of Pincodes flagged as 'High Risk'.
            
            **Why this matters:**
            Fraud is often geographically concentrated due to local operators. This chart highlights the **"Hotspots"**—districts where the Ghost Village phenomenon is systemic rather than isolated.
            """)

        # Filter for High Risk only for the bar chart to see hotspots
        risk_only = df_analyzed[df_analyzed['Risk_Profile'] == 'High Risk (Ghost Village)']
        if not risk_only.empty:
            district_risk = risk_only.groupby('district')['pincode'].count().reset_index().sort_values('pincode', ascending=False).head(15)
            fig_bar = px.bar(
                district_risk,
                x='district',
                y='pincode',
                title="Top Districts by Number of 'Ghost' Pincodes",
                color='pincode',
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)", 
                font_color="#e0f7fa",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No High Risk districts found in current filter selection.")

    with c2:
        st.subheader("3. Risk Ratio")
        
        with st.expander("ℹ️ Guide: Pie Chart"):
            st.markdown("""
            **Type:** **Univariate Analysis**.
            Shows the proportional composition of the dataset by Risk Profile.
            """)

        fig_pie = px.pie(
            df_analyzed, 
            names='Risk_Profile', 
            title='Percentage of Pincodes by Risk Category',
            color='Risk_Profile',
            color_discrete_map={
            "High Risk (Ghost Village)": "#ff5252",
            "Medium Risk (Monitor)": "#ffd740",
            "Low Risk (Normal Activity)": "#69f0ae"
            },
            hole=0.4
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0f7fa")
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Advanced Visualization Deep Dives ---
    st.markdown("---")
    st.header("4. Advanced Forensic Deep Dives")
    st.markdown("Detailed multi-dimensional analysis to isolate specific fraud patterns.")

    tab1, tab2, tab3, tab4 = st.tabs(["3D Cluster View", "Geographic Hierarchy", "Risk Flow Analysis", "Statistical Deviations"])

    with tab1:
        st.subheader("Multi-Dimensional Outlier Analysis")
        with st.expander("ℹ️ Visualization Guide: 3D Scatter (Multivariate Analysis)"):
            st.markdown("""
            **Type of Analysis:** **Multivariate Analysis** (3 Variables).
            * **X-Axis:** Total Enrolments.
            * **Y-Axis:** Biometric Updates.
            * **Z-Axis:** Demographic Updates.
            
            **Deep Insight:** Standard 2D plots group all updates together. This 3D view allows us to spot **"Partial Ghosts"**—areas that might be faking demographic updates (easier to forge) but have zero biometric updates (harder to forge).
            """)
        
        fig_3d = px.scatter_3d(
            df_analyzed,
            x='total_enrolments',
            y='total_bio_updates',
            z='total_demo_updates',
            color='Risk_Profile',
            size='total_enrolments',
            hover_data=['pincode', 'district'],
            color_discrete_map={
                "High Risk (Ghost Village)": "#ff5252",
                "Medium Risk (Monitor)": "#ffd740",
                "Low Risk (Normal Activity)": "#69f0ae"
            },
            title="3D Forensic Scatter: Enrolment vs Bio vs Demo"
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Enrolments',
                yaxis_title='Biometric Updates',
                zaxis_title='Demographic Updates',
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)', showbackground=True),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)', showbackground=True),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor='rgba(255,255,255,0.1)', showbackground=True),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0f7fa",
            height=700
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab2:
        st.subheader("Hierarchical Fraud Detection")
        with st.expander("ℹ️ Visualization Guide: Treemap (Hierarchical Multivariate Analysis)"):
            st.markdown("""
            **Type of Analysis:** **Hierarchical Multivariate Analysis**.
            * **Hierarchy:** Country → State → District.
            * **Size:** Total Enrolment Volume (Quantitative).
            * **Color:** Risk Profile (Categorical).
            
            **Deep Insight:**
            This helps identify **Regional Contagion**. If a specific State box is overwhelmingly Red, it indicates a policy-level or state-level systemic issue rather than isolated operator fraud.
            """)
        
        # To make the treemap readable, we limit depth or aggregate if too large
        # We'll use a copy to ensure 'India' root node
        tree_df = df_analyzed.copy()
        tree_df["Country"] = "India"
        
        fig_tree = px.treemap(
            tree_df,
            path=['Country', 'state', 'district'],
            values='total_enrolments',
            color='Risk_Profile',
            color_discrete_map={
                "High Risk (Ghost Village)": "#ff5252",
                "Medium Risk (Monitor)": "#ffd740",
                "Low Risk (Normal Activity)": "#69f0ae",
                "(?)": "#262730"
            },
            title="Geographic Treemap: Size = Enrolment Volume, Color = Risk"
        )
        fig_tree.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0f7fa", height=600)
        st.plotly_chart(fig_tree, use_container_width=True)

    with tab3:
        st.subheader("Risk Flow: State to Profile")
        with st.expander("ℹ️ Visualization Guide: Parallel Categories (Multivariate Categorical Flow)"):
            st.markdown("""
            **Type of Analysis:** **Multivariate Categorical Analysis**.
            * **Dimensions:** State → Risk Profile.
            
            **Deep Insight:**
            This visualization acts like a **Sankey Diagram**. It shows the 'flow' of data. You can instantly see which State contributes the thickest 'stream' to the Red 'High Risk' bar. It normalizes the view to show **proportional contribution**.
            """)
        
        # Group data for Parallel Categories to avoid overcrowding
        cat_df = df_analyzed.groupby(['state', 'Risk_Profile']).size().reset_index(name='count')
        # Filter top states by volume if too many
        top_states = cat_df.groupby('state')['count'].sum().nlargest(10).index
        cat_df = cat_df[cat_df['state'].isin(top_states)]
        
        fig_sankey = px.parallel_categories(
            cat_df,
            dimensions=['state', 'Risk_Profile'],
            color='count',
            color_continuous_scale=px.colors.sequential.Inferno,
            title="Flow Analysis: Which States Contribute Most to High Risk?"
        )
        fig_sankey.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0f7fa", height=500)
        st.plotly_chart(fig_sankey, use_container_width=True)

    with tab4:
        st.subheader("Statistical Validation")
        with st.expander("ℹ️ Visualization Guide: Violin Plot (Bivariate Statistical Analysis)"):
            st.markdown("""
            **Type of Analysis:** **Bivariate Analysis** (Numerical Distribution vs Categorical).
            * **X-Axis:** Risk Category.
            * **Y-Axis:** Total Updates (Log Scale).
            * **Shape:** Probability Density.
            
            **Deep Insight:**
            Violin plots show the **shape** of the data. 
            * **High Risk (Red):** Should look like a flat line or a bulge near zero (indicating most data points have ~0 updates).
            * **Low Risk (Green):** Should look like a long violin extending upwards (indicating a healthy variety of update counts).
            This statistically proves that the clusters are distinct populations.
            """)
        
        fig_violin = px.violin(
            df_analyzed,
            y="total_updates",
            x="Risk_Profile",
            box=True,
            points="all",
            color="Risk_Profile",
            color_discrete_map={
                "High Risk (Ghost Village)": "#ff5252",
                "Medium Risk (Monitor)": "#ffd740",
                "Low Risk (Normal Activity)": "#69f0ae"
            },
            title="Distribution of Updates across Risk Profiles"
        )
        fig_violin.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0f7fa", yaxis_type="log")
        st.plotly_chart(fig_violin, use_container_width=True)


    # --- Audit List Generation ---
    st.markdown("---")
    st.subheader("📋 Physical Verification Audit List")
    with st.expander("ℹ️ About the Audit List Generation"):
        st.markdown("""
        **Methodology:**
        We calculate a **'Suspicion Score'** for every pincode in the High Risk Cluster.
        $$ Suspicion Score = \\frac{Total Enrolments \\times 10}{Total Updates + 1} $$
        
        This formula prioritizes locations with **High Volume** AND **Low Updates**. A pincode with 10,000 enrolments and 0 updates will have a higher score than a pincode with 100 enrolments and 0 updates, ensuring auditors focus on the biggest potential scams first.
        """)

    st.markdown("These locations require immediate physical verification. The 'Suspicion Score' is calculated as the inverse of the update ratio.")

    # Create a clean view for the table
    audit_df = df_analyzed[df_analyzed['Risk_Profile'] == 'High Risk (Ghost Village)'].copy()
    
    if not audit_df.empty:
        # Calculate a simplified "Suspicion Score" for sorting (High Enrolment + Low Updates = Higher Score)
        audit_df['Suspicion_Score'] = (audit_df['total_enrolments'] * 10) / (audit_df['total_updates'] + 1)
        audit_df = audit_df.sort_values('Suspicion_Score', ascending=False)
        
        display_cols = ['state', 'district', 'pincode', 'total_enrolments', 'total_updates', 'Risk_Profile', 'Suspicion_Score']
        
        st.dataframe(
            audit_df[display_cols].style.background_gradient(subset=['Suspicion_Score'], cmap='Reds'),
            use_container_width=True
        )

        # Download Button
        csv = audit_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Audit Target List (CSV)",
            data=csv,
            file_name='ghost_village_audit_targets.csv',
            mime='text/csv',
        )
    else:
        st.success("Analysis complete. No 'Ghost Villages' detected with current filter parameters.")

if __name__ == "__main__":
    main()