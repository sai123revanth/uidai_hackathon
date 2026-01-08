import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

# --- 1. SEO & PAGE CONFIGURATION ---
st.set_page_config(
    page_title="India Demographic Dividend | Education & Workforce Analytics",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.uidai.gov.in/',
        'Report a bug': "mailto:admin@example.com",
        'About': "# Demographic Dividend Dashboard\nThis tool visualizes Aadhar update trends to identify education and workforce hubs."
    }
)

# --- 2. META TAGS & RESPONSIVE STYLING ---
seo_meta_tags = """
<div style="display: none;">
    <h1>India Demographic Dividend Dashboard</h1>
    <p>Analyze district-level demographic data to distinguish between young education hubs and mature workforce engines.</p>
</div>
<style>
    /* Professional Dark Gradient Background */
    .stApp {
        background: radial-gradient(circle at top center, #1e293b 0%, #0f172a 100%);
        background-attachment: fixed;
    }

    /* GLASSMORPHISM CARDS */
    .metric-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(5px);
    }
    
    h1, h2, h3 { 
        font-family: 'Helvetica Neue', sans-serif; 
        color: #f1f5f9;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(15, 23, 42, 0.6);
        border-radius: 5px;
        color: #cbd5e1;
        padding: 10px 20px;
        border: 1px solid #1e293b;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 41, 59, 0.8);
        border-bottom: 2px solid #00CC96;
        color: white;
    }

    /* --- MOBILE OPTIMIZATION --- */
    /* Adjusts layout for screens smaller than 768px (Tablets/Phones) */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        h1 {
            font-size: 1.5rem !important;
        }
        /* Make charts taller on mobile for better visibility */
        .js-plotly-plot {
            height: 400px !important;
        }
    }
</style>
"""
st.markdown(seo_meta_tags, unsafe_allow_html=True)

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    file_pattern = "api_data_aadhar_demographic_*.csv"
    files = glob.glob(file_pattern)
    
    if not files:
        files = glob.glob("*.csv")
    
    if not files:
        return None, None

    df_list = []
    for file in files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
            continue
            
    if not df_list:
        return None, None
        
    raw_df = pd.concat(df_list, ignore_index=True)
    
    # Cleaning
    cols = raw_df.columns
    youth_col = 'demo_age_5_17'
    adult_col = [c for c in cols if c.startswith('demo_age_17')][0] 
    
    raw_df = raw_df.rename(columns={
        youth_col: 'Youth_Updates',
        adult_col: 'Adult_Updates',
        'state': 'State',
        'district': 'District',
        'date': 'Date'
    })
    
    raw_df['Youth_Updates'] = pd.to_numeric(raw_df['Youth_Updates'], errors='coerce').fillna(0)
    raw_df['Adult_Updates'] = pd.to_numeric(raw_df['Adult_Updates'], errors='coerce').fillna(0)
    raw_df['Date'] = pd.to_datetime(raw_df['Date'], format='%d-%m-%Y', errors='coerce')

    # Aggregations
    district_df = raw_df.groupby(['State', 'District'])[['Youth_Updates', 'Adult_Updates']].sum().reset_index()
    district_df['Total_Updates'] = district_df['Youth_Updates'] + district_df['Adult_Updates']
    district_df['Youth_Index'] = (district_df['Youth_Updates'] / district_df['Total_Updates']) * 100
    
    raw_df['Month_Year'] = raw_df['Date'].dt.to_period('M').astype(str)
    trend_df = raw_df.groupby(['Month_Year'])[['Youth_Updates', 'Adult_Updates']].sum().reset_index()
    trend_df['Total_Updates'] = trend_df['Youth_Updates'] + trend_df['Adult_Updates']
    
    return district_df, trend_df

# Load Data
district_df_full, trend_df_full = load_data()

if district_df_full is None:
    st.error("No data found. Please place the CSV files in the same directory.")
    st.stop()

# --- 3. DEEP LINKING ---
query_params = st.query_params
default_states = query_params.get_all("state") if "state" in query_params else []
valid_states = sorted(district_df_full['State'].unique())
default_states = [s for s in default_states if s in valid_states]

# --- SIDEBAR: FILTERS ---
st.sidebar.title("🔍 Analytic Filters")

selected_states = st.sidebar.multiselect(
    "Filter by State",
    options=valid_states,
    default=default_states,
    help="Select one or more states. URL updates automatically."
)

if selected_states:
    st.query_params["state"] = selected_states
else:
    if "state" in st.query_params:
        del st.query_params["state"]

min_updates = st.sidebar.slider(
    "Minimum Data Volume (Noise Filter)",
    min_value=0,
    max_value=int(district_df_full['Total_Updates'].quantile(0.9)), 
    value=100,
    step=50,
    help="Filter out districts with insignificant data volume."
)

filtered_df = district_df_full[district_df_full['Total_Updates'] >= min_updates]
if selected_states:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

# --- MAIN DASHBOARD CONTENT ---
st.title("🇮🇳 National Demographic Intelligence")
st.markdown("### Strategic Analysis: Education Hubs vs. Workforce Engines")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", 
    "🗺️ State Analytics", 
    "📈 Trend Analysis", 
    "🔎 District Explorer"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY
# ==========================================
with tab1:
    total_vol = filtered_df['Total_Updates'].sum()
    avg_index = filtered_df['Youth_Index'].mean()
    
    if not filtered_df.empty:
        youngest = filtered_df.loc[filtered_df['Youth_Index'].idxmax()]
        maturest = filtered_df.loc[filtered_df['Youth_Index'].idxmin()]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data Points", f"{total_vol:,.0f}")
        c2.metric("Avg Youth Index", f"{avg_index:.1f}%")
        c3.metric("Youngest Dist.", f"{youngest['District']}") # Shortened title for mobile
        c4.metric("Maturest Dist.", f"{maturest['District']}") # Shortened title for mobile
    else:
        st.warning("No data meets the current filter criteria.")

    st.markdown("---")
    
    st.subheader("📍 The Demographic Heatmap")
    
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
    
    st.markdown("### 🏛️ Policy Recommendations")
    ac1, ac2 = st.columns(2)
    
    # We select specific columns to avoid horizontal scroll on mobile
    cols_to_show = ['State', 'District', 'Youth_Index', 'Total_Updates']
    
    with ac1:
        st.info("🎒 **Education Priority**")
        top_youth = filtered_df.nlargest(10, 'Youth_Index')
        st.dataframe(
            top_youth[cols_to_show].style.format({"Youth_Index": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

    with ac2:
        st.error("💼 **Workforce Priority**")
        top_work = filtered_df.nsmallest(10, 'Youth_Index').copy()
        top_work['Adult_Index'] = 100 - top_work['Youth_Index']
        # Show adult index instead of youth index for this table
        st.dataframe(
            top_work[['State', 'District', 'Adult_Index', 'Total_Updates']].style.format({"Adult_Index": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# TAB 2: STATE ANALYTICS
# ==========================================
with tab2:
    st.subheader("State-Level Comparative Analysis")
    
    state_stats = filtered_df.groupby('State').agg({
        'Total_Updates': 'sum',
        'Youth_Updates': 'sum',
        'Adult_Updates': 'sum',
        'Youth_Index': 'mean' 
    }).reset_index().sort_values('Youth_Index', ascending=False)
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        fig_state = px.bar(
            state_stats,
            x='State',
            y='Youth_Index',
            color='Youth_Index',
            color_continuous_scale='RdYlGn',
            title="Youngest to Oldest States",
            labels={'Youth_Index': 'Avg Youth Index (%)'},
            height=500
        )
        fig_state.add_hline(y=state_stats['Youth_Index'].mean(), line_dash="dot", annotation_text="Avg")
        st.plotly_chart(fig_state, use_container_width=True)
        
    with col_b:
        st.write("### State Statistics")
        st.dataframe(
            state_stats[['State', 'Total_Updates', 'Youth_Index']],
            column_config={
                "Youth_Index": st.column_config.ProgressColumn(
                    "Youth Index",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Total_Updates": st.column_config.NumberColumn(
                    "Volume",
                    format="%d"
                )
            },
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# TAB 3: TREND ANALYSIS
# ==========================================
with tab3:
    st.subheader("📅 Temporal Trends")
    
    if trend_df_full is not None and not trend_df_full.empty:
        trend_df_full = trend_df_full.sort_values('Month_Year')
        
        fig_trend = px.area(
            trend_df_full, 
            x='Month_Year', 
            y=['Youth_Updates', 'Adult_Updates'],
            title="Monthly Update Volume Trends",
            labels={'value': 'Updates', 'variable': 'Segment'},
            color_discrete_map={'Youth_Updates': '#00CC96', 'Adult_Updates': '#EF553B'}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Not enough temporal data available to show trends.")

# ==========================================
# TAB 4: DISTRICT EXPLORER
# ==========================================
with tab4:
    st.subheader("🔎 District Deep Dive")
    
    c_search, c_chart = st.columns([1, 2])
    
    with c_search:
        st.markdown("### Find a District")
        search_term = st.text_input("Search Name", placeholder="e.g., Mahabubnagar")
        
        display_df = filtered_df.copy()
        if search_term:
            display_df = display_df[display_df['District'].str.contains(search_term, case=False)]
            
        fig_hist = px.histogram(
            filtered_df, 
            x="Youth_Index", 
            nbins=30, 
            title="Distribution",
            color_discrete_sequence=['#636EFA']
        )
        fig_hist.update_layout(showlegend=False, height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

    with c_chart:
        fig_scatter = px.scatter(
            display_df,
            x='Total_Updates',
            y='Youth_Index',
            size='Total_Updates',
            color='State',
            hover_name='District',
            log_x=True,
            title="Volume vs Youth Index",
            height=500
        )
        fig_scatter.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### 📋 Comprehensive District Data")
    # Reduced columns for Mobile View - Removed raw counts, kept Key Metrics
    mobile_friendly_cols = ['District', 'State', 'Total_Updates', 'Youth_Index']
    
    st.dataframe(
        display_df[mobile_friendly_cols].sort_values(by='Total_Updates', ascending=False),
        column_config={
            "Youth_Index": st.column_config.ProgressColumn(
                "Youth Index",
                help="Percentage of updates from Age 5-17",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Total_Updates": st.column_config.NumberColumn(
                "Total Vol",
                format="%d"
            )
        },
        use_container_width=True,
        hide_index=True
    )