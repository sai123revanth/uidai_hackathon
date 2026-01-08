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
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    .highlight {
        color: #00CC96; /* Teal for Youth */
    }
    .mature {
        color: #EF553B; /* Red/Orange for Workforce */
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
        # Fallback if no specific files found, try any csv in dir
        files = glob.glob("*.csv")
    
    if not files:
        return None

    df_list = []
    for file in files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
            continue
            
    if not df_list:
        return None
        
    raw_df = pd.concat(df_list, ignore_index=True)
    
    # --- DATA CLEANING ---
    # Identify columns. We expect 'demo_age_5_17' and 'demo_age_17_' (or similar)
    # The snippet showed 'demo_age_17_', likely truncated. We'll find the column that starts with 'demo_age_17'
    
    cols = raw_df.columns
    youth_col = 'demo_age_5_17'
    adult_col = [c for c in cols if c.startswith('demo_age_17')][0] # Auto-detect 17+ column
    
    # Rename for clarity
    raw_df = raw_df.rename(columns={
        youth_col: 'Youth_Updates',
        adult_col: 'Adult_Updates',
        'state': 'State',
        'district': 'District'
    })
    
    # Ensure numeric
    raw_df['Youth_Updates'] = pd.to_numeric(raw_df['Youth_Updates'], errors='coerce').fillna(0)
    raw_df['Adult_Updates'] = pd.to_numeric(raw_df['Adult_Updates'], errors='coerce').fillna(0)
    
    # Group by District and State
    # aggregating sum of updates across all dates/pincodes for the district
    district_df = raw_df.groupby(['State', 'District'])[['Youth_Updates', 'Adult_Updates']].sum().reset_index()
    
    # --- METRIC CALCULATIONS ---
    district_df['Total_Updates'] = district_df['Youth_Updates'] + district_df['Adult_Updates']
    
    # Youth Index: Percentage of updates that are from children (5-17)
    # Formula: (Youth / Total) * 100
    district_df['Youth_Index'] = (district_df['Youth_Updates'] / district_df['Total_Updates']) * 100
    
    # Filter out low-data noise (districts with very few updates might skew ratios)
    # Only consider districts with at least significant volume (e.g. > 100 updates in sample)
    district_df = district_df[district_df['Total_Updates'] > 100]
    
    return district_df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.title("🔍 Filters")

if df is not None:
    selected_states = st.sidebar.multiselect(
        "Select States (Leave empty for All)",
        options=sorted(df['State'].unique())
    )
    
    filtered_df = df[df['State'].isin(selected_states)] if selected_states else df
else:
    st.error("No data found. Please place the CSV files in the same directory.")
    st.stop()

# --- HEADER SECTION ---
st.title("🇮🇳 Demographic Dividend: Policy Intelligence")
st.markdown("### The Insight: distinguishing 'Young' districts (Education priority) from 'Mature' districts (Job priority).")
st.markdown("---")

# --- KPI ROW ---
col1, col2, col3, col4 = st.columns(4)

total_volume = filtered_df['Total_Updates'].sum()
avg_youth_index = filtered_df['Youth_Index'].mean()

# Identify youngest and oldest districts in the filtered set
youngest_district = filtered_df.loc[filtered_df['Youth_Index'].idxmax()]
oldest_district = filtered_df.loc[filtered_df['Youth_Index'].idxmin()]

with col1:
    st.metric("Total Data Points", f"{total_volume:,.0f}", help="Total demographic updates analyzed")
with col2:
    st.metric("Avg Youth Index", f"{avg_youth_index:.1f}%", help="Average % of updates from Age 5-17 across selected region")
with col3:
    st.metric("Youngest District", f"{youngest_district['District']}", f"{youngest_district['Youth_Index']:.1f}% Youth", delta_color="normal")
with col4:
    st.metric("Most Mature District", f"{oldest_district['District']}", f"{(100-oldest_district['Youth_Index']):.1f}% Adult", delta_color="inverse")

st.markdown("---")

# --- MAIN VISUALIZATION: TREEMAP ---
# A Treemap is better than a Choropleth here because it shows Volume (Size) AND Index (Color) simultaneously
# and doesn't rely on potentially messy district name matching with GeoJSONs.

st.subheader("🗺️ The National Demographic Landscape")
st.caption("Size = Total Volume of Updates | Color = Youth Index (Brighter/Yellow = More Kids, Darker/Blue = More Adults)")

fig_treemap = px.treemap(
    filtered_df,
    path=[px.Constant("India"), 'State', 'District'],
    values='Total_Updates',
    color='Youth_Index',
    color_continuous_scale='Viridis', # Yellow (Young) to Blue (Mature)
    hover_data=['Youth_Updates', 'Adult_Updates'],
    title="Demographic Heatmap: Drilling down from State to District"
)
fig_treemap.update_layout(margin=dict(t=30, l=10, r=10, b=10), height=600)
st.plotly_chart(fig_treemap, use_container_width=True)


# --- INSIGHTS SECTION: THE JURY'S ACTION PLAN ---
st.markdown("## 🏛️ Policy Recommendations")

c1, c2 = st.columns(2)

# --- LEFT: EDUCATION HUBS ---
with c1:
    st.markdown("### 🎒 The Education Hubs (High Youth Ratio)")
    st.markdown("""
    **Insight:** These districts have a disproportionately high number of updates from children (5-17).
    \n**Action:** Prioritize **scholarship disbursement**, **new school centers**, and **digital literacy programs** here.
    """)
    
    top_youth = filtered_df.nlargest(10, 'Youth_Index')[['State', 'District', 'Youth_Index', 'Total_Updates']]
    
    fig_youth = px.bar(
        top_youth,
        x='Youth_Index',
        y='District',
        orientation='h',
        color='Youth_Index',
        color_continuous_scale='Greens',
        text_auto='.1f',
        title="Top 10 Districts: Highest Child Density"
    )
    fig_youth.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig_youth, use_container_width=True)

# --- RIGHT: WORKFORCE ENGINES ---
with c2:
    st.markdown("### 💼 The Workforce Engines (High Adult Ratio)")
    st.markdown("""
    **Insight:** These districts are dominated by the 17+ workforce demographic.
    \n**Action:** Prioritize **job fairs**, **banking/credit expansion**, and **upskilling centers** here.
    """)
    
    # Low Youth Index = High Adult Index
    top_workforce = filtered_df.nsmallest(10, 'Youth_Index')[['State', 'District', 'Youth_Index', 'Total_Updates']]
    # Create a "Mature Index" for display
    top_workforce['Mature_Index'] = 100 - top_workforce['Youth_Index']
    
    fig_work = px.bar(
        top_workforce,
        x='Mature_Index',
        y='District',
        orientation='h',
        color='Mature_Index',
        color_continuous_scale='Reds',
        text_auto='.1f',
        title="Top 10 Districts: Highest Workforce Density"
    )
    fig_work.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig_work, use_container_width=True)


# --- DEEP DIVE SCATTER PLOT ---
st.markdown("---")
st.subheader("📊 Strategic Quadrants: Volume vs. Youth Index")
st.markdown("Use this to find **High Volume opportunities**. Look for large circles (high volume) in the top-right (Young & Active).")

fig_scatter = px.scatter(
    filtered_df,
    x='Total_Updates',
    y='Youth_Index',
    size='Total_Updates',
    color='State',
    hover_name='District',
    log_x=True, # Log scale helps view data with vast differences in volume
    title="District Distribution: Identifying Outliers",
    labels={'Youth_Index': 'Youth Index (%)', 'Total_Updates': 'Total Updates (Log Scale)'}
)

# Add reference lines
avg_y = filtered_df['Youth_Index'].mean()
fig_scatter.add_hline(y=avg_y, line_dash="dash", line_color="white", annotation_text="National Avg")

st.plotly_chart(fig_scatter, use_container_width=True)

# --- RAW DATA VIEW ---
with st.expander("📂 View Raw Data Analysis"):
    st.dataframe(filtered_df.sort_values(by='Total_Updates', ascending=False))