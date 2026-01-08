import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Demographic Dividend: Education vs Workforce",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0E1117;
        border-radius: 5px;
        color: white;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        border-bottom: 2px solid #00CC96;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    # Pattern to match your specific uploaded files
    file_pattern = "api_data_aadhar_demographic_*.csv"
    files = glob.glob(file_pattern)
    
    if not files:
        files = glob.glob("*.csv")
    
    if not files:
        return None, None

    df_list = []
    for file in files:
        try:
            # Read only necessary columns to save memory if needed, but here we read all
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
            continue
            
    if not df_list:
        return None, None
        
    raw_df = pd.concat(df_list, ignore_index=True)
    
    # --- DATA CLEANING ---
    cols = raw_df.columns
    youth_col = 'demo_age_5_17'
    # Auto-detect 17+ column (usually 'demo_age_17_')
    adult_col = [c for c in cols if c.startswith('demo_age_17')][0] 
    
    # Rename for clarity
    raw_df = raw_df.rename(columns={
        youth_col: 'Youth_Updates',
        adult_col: 'Adult_Updates',
        'state': 'State',
        'district': 'District',
        'date': 'Date'
    })
    
    # Convert types
    raw_df['Youth_Updates'] = pd.to_numeric(raw_df['Youth_Updates'], errors='coerce').fillna(0)
    raw_df['Adult_Updates'] = pd.to_numeric(raw_df['Adult_Updates'], errors='coerce').fillna(0)
    
    # Parse Date (assuming dd-mm-yyyy format from snippets)
    raw_df['Date'] = pd.to_datetime(raw_df['Date'], format='%d-%m-%Y', errors='coerce')

    # --- AGGREGATION 1: DISTRICT LEVEL (For Maps/Scatter) ---
    district_df = raw_df.groupby(['State', 'District'])[['Youth_Updates', 'Adult_Updates']].sum().reset_index()
    district_df['Total_Updates'] = district_df['Youth_Updates'] + district_df['Adult_Updates']
    district_df['Youth_Index'] = (district_df['Youth_Updates'] / district_df['Total_Updates']) * 100
    
    # --- AGGREGATION 2: TIME SERIES (For Trends) ---
    # Group by Month and Year to reduce data size for plotting
    raw_df['Month_Year'] = raw_df['Date'].dt.to_period('M').astype(str)
    trend_df = raw_df.groupby(['Month_Year'])[['Youth_Updates', 'Adult_Updates']].sum().reset_index()
    trend_df['Total_Updates'] = trend_df['Youth_Updates'] + trend_df['Adult_Updates']
    
    return district_df, trend_df

# Load Data
district_df_full, trend_df_full = load_data()

if district_df_full is None:
    st.error("No data found. Please place the CSV files in the same directory.")
    st.stop()

# --- SIDEBAR: ADVANCED FILTERS ---
st.sidebar.title("🔍 Advanced Filters")

# 1. State Filter
selected_states = st.sidebar.multiselect(
    "Filter by State",
    options=sorted(district_df_full['State'].unique()),
    help="Select one or more states to analyze specific regions."
)

# 2. Data Volume Noise Filter
min_updates = st.sidebar.slider(
    "Minimum Data Volume (Noise Filter)",
    min_value=0,
    max_value=int(district_df_full['Total_Updates'].quantile(0.9)), # Cap at 90th percentile
    value=100,
    step=50,
    help="Filter out districts with very few updates to ensure statistical significance."
)

# Apply Filters
filtered_df = district_df_full[district_df_full['Total_Updates'] >= min_updates]
if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

# --- MAIN DASHBOARD STRUCTURE ---
st.title("🇮🇳 National Demographic Intelligence Dashboard")
st.markdown("### Strategic Analysis of Youth vs. Workforce Distribution")

# Create Tabs for organized view
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "🗺️ State Analytics", 
    "📈 Trend Analysis", 
    "🔎 District Explorer"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY (The "Jury" View)
# ==========================================
with tab1:
    # KPI Metrics
    total_vol = filtered_df['Total_Updates'].sum()
    avg_index = filtered_df['Youth_Index'].mean()
    
    # Find outliers
    if not filtered_df.empty:
        youngest = filtered_df.loc[filtered_df['Youth_Index'].idxmax()]
        maturest = filtered_df.loc[filtered_df['Youth_Index'].idxmin()]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data Points", f"{total_vol:,.0f}")
        c2.metric("Avg Youth Index", f"{avg_index:.1f}%")
        c3.metric("Youngest District", f"{youngest['District']}", f"{youngest['Youth_Index']:.1f}% Youth")
        c4.metric("Most Mature District", f"{maturest['District']}", f"{100-maturest['Youth_Index']:.1f}% Adult")
    else:
        st.warning("No data meets the current filter criteria.")

    st.markdown("---")
    
    # Treemap (The Big Picture)
    st.subheader("📍 The Demographic Map")
    st.caption("Visualizing the hierarchy: Region > State > District. Color indicates Age Profile.")
    
    fig_tree = px.treemap(
        filtered_df,
        path=[px.Constant("India"), 'State', 'District'],
        values='Total_Updates',
        color='Youth_Index',
        color_continuous_scale='Viridis',
        hover_data=['Youth_Updates', 'Adult_Updates'],
        height=500
    )
    st.plotly_chart(fig_tree, use_container_width=True)
    
    # Actionable Recommendations
    st.markdown("### 🏛️ Policy Action Plan")
    ac1, ac2 = st.columns(2)
    
    with ac1:
        st.info("🎒 **Education Priority Zones (High Youth Index)**")
        st.markdown("*Districts needing scholarships, schools, and digital literacy.*")
        top_youth = filtered_df.nlargest(10, 'Youth_Index')
        st.dataframe(
            top_youth[['State', 'District', 'Youth_Index', 'Total_Updates']].style.format({"Youth_Index": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

    with ac2:
        st.error("💼 **Workforce Priority Zones (High Adult Index)**")
        st.markdown("*Districts needing job fairs, credit access, and upskilling.*")
        top_work = filtered_df.nsmallest(10, 'Youth_Index').copy()
        top_work['Adult_Index'] = 100 - top_work['Youth_Index']
        st.dataframe(
            top_work[['State', 'District', 'Adult_Index', 'Total_Updates']].style.format({"Adult_Index": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# TAB 2: STATE ANALYTICS (Comparative View)
# ==========================================
with tab2:
    st.subheader("State-Level Performance Comparison")
    
    # Group by State for high-level comparison
    state_stats = filtered_df.groupby('State').agg({
        'Total_Updates': 'sum',
        'Youth_Updates': 'sum',
        'Adult_Updates': 'sum',
        'Youth_Index': 'mean' # Average of district indices
    }).reset_index()
    
    # Sort by Youth Index
    state_stats = state_stats.sort_values('Youth_Index', ascending=False)
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        fig_state = px.bar(
            state_stats,
            x='State',
            y='Youth_Index',
            color='Youth_Index',
            color_continuous_scale='RdYlGn', # Red (Old) to Green (Young)
            title="Which States have the youngest demographic profile?",
            labels={'Youth_Index': 'Avg Youth Index (%)'},
            height=500
        )
        fig_state.add_hline(y=state_stats['Youth_Index'].mean(), line_dash="dot", annotation_text="National Avg")
        st.plotly_chart(fig_state, use_container_width=True)
        
    with col_b:
        st.write("### State Statistics Table")
        st.dataframe(
            state_stats[['State', 'Total_Updates', 'Youth_Index']].style.background_gradient(cmap='Greens', subset=['Youth_Index']),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# TAB 3: TREND ANALYSIS (Time Series)
# ==========================================
with tab3:
    st.subheader("📅 Demographic Updates Over Time")
    
    if trend_df_full is not None and not trend_df_full.empty:
        # Sort by date naturally
        trend_df_full = trend_df_full.sort_values('Month_Year')
        
        # Line Chart: Volume
        fig_trend = px.area(
            trend_df_full, 
            x='Month_Year', 
            y=['Youth_Updates', 'Adult_Updates'],
            title="Monthly Update Volume: Youth vs Adults",
            labels={'value': 'Number of Updates', 'variable': 'Demographic'},
            color_discrete_map={'Youth_Updates': '#00CC96', 'Adult_Updates': '#EF553B'}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.caption("Note: Trends are aggregated nationally. Filtering by State in the sidebar currently affects the District/State tabs, but this global trend view remains national to show overall data intake health.")
    else:
        st.warning("Not enough temporal data available to show trends.")

# ==========================================
# TAB 4: DISTRICT EXPLORER (Deep Dive)
# ==========================================
with tab4:
    st.subheader("🔎 Deep Dive: District Explorer")
    
    c_search, c_chart = st.columns([1, 2])
    
    with c_search:
        st.markdown("### Search & Filter")
        # Text Search
        search_term = st.text_input("Search for a District", placeholder="e.g., Mahabubnagar")
        
        # Filter Logic for Table
        display_df = filtered_df.copy()
        if search_term:
            display_df = display_df[display_df['District'].str.contains(search_term, case=False)]
            
        # Histogram of Distribution
        fig_hist = px.histogram(
            filtered_df, 
            x="Youth_Index", 
            nbins=30, 
            title="Distribution of Youth Index",
            color_discrete_sequence=['#636EFA']
        )
        fig_hist.update_layout(showlegend=False, height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_chart:
        # Scatter Plot (Volume vs Index)
        fig_scatter = px.scatter(
            display_df,
            x='Total_Updates',
            y='Youth_Index',
            size='Total_Updates',
            color='State',
            hover_name='District',
            log_x=True,
            title="Strategic Quadrants: Volume vs Youth Index",
            height=500
        )
        fig_scatter.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Detailed Table
    st.markdown("### Detailed District Data")
    st.dataframe(
        display_df.sort_values(by='Total_Updates', ascending=False),
        column_config={
            "Youth_Index": st.column_config.ProgressColumn(
                "Youth Index",
                help="Percentage of updates from Age 5-17",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Total_Updates": st.column_config.NumberColumn(
                "Total Volume",
                format="%d"
            )
        },
        use_container_width=True,
        hide_index=True
    )