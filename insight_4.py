import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Migration Hotspots & Service Prediction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Styling ---
# Updated to Navy Blue Theme for better visibility and aesthetics
st.markdown("""
<style>
    /* Main App Background - Navy Blue */
    .stApp {
        background-color: #0a192f;
        color: #ffffff;
    }
    
    /* Text Visibility Correction */
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #e6f1ff; 
    }
    
    /* Specific override for Streamlit metrics to ensure visibility */
    [data-testid="stMetricLabel"] {
        color: #8892b0 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricDelta"] {
        color: #64ffda !important;
    }

    /* Custom Metric Card Styling */
    .metric-card {
        background-color: #112240; /* Lighter Navy for contrast */
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #233554;
    }
    .metric-title {
        color: #8892b0; /* Slate color for titles */
        font-size: 14px;
        font-weight: bold;
    }
    .metric-value {
        color: #ccd6f6; /* Lightest slate for values */
        font-size: 24px;
        font-weight: bold;
    }
    
    /* Insight Box Styling */
    .insight-box {
        background-color: #172a45; /* Distinct dark blue */
        border-left: 5px solid #64ffda; /* Teal accent */
        padding: 15px;
        margin-bottom: 20px;
        color: #ffffff;
        border-radius: 0 5px 5px 0;
    }
    
    /* Explanation Text Styling */
    .explanation-text {
        font-size: 14px;
        color: #a8b2d1; /* Softer text color for reading */
        background-color: #112240;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #233554;
        margin-top: 10px;
    }
    
    /* Recommendation Header */
    .recommendation-header {
        color: #64ffda; /* Teal for positive actions */
        font-weight: bold;
        margin-top: 10px;
    }
    
    /* Tab Styling adjustments */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #112240;
        border-radius: 4px;
        color: #8892b0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #64ffda;
        color: #0a192f;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading Function ---
@st.cache_data
def load_and_process_data():
    # 1. Load Biometric Data (Adding new files to the list)
    bio_files = [
        "api_data_aadhar_biometric_0_500000.csv",
        "api_data_aadhar_biometric_500000_1000000.csv",
        "api_data_aadhar_biometric_1000000_1500000.csv",
        "api_data_aadhar_biometric_1500000_1861108.csv"
    ]
    df_bio_list = []
    for f in bio_files:
        try:
            df_bio_list.append(pd.read_csv(f))
        except FileNotFoundError:
            continue
    
    if df_bio_list:
        df_bio = pd.concat(df_bio_list)
        # Rename columns to standardized format
        df_bio.rename(columns={
            'bio_age_5_17': 'Biometric_5_17',
            'bio_age_17_': 'Biometric_18_plus'
        }, inplace=True)
    else:
        df_bio = pd.DataFrame(columns=['date', 'state', 'district', 'pincode', 'Biometric_5_17', 'Biometric_18_plus'])

    # 2. Load Demographic Data (Adding new files to the list)
    demo_files = [
        "api_data_aadhar_demographic_1000000_1500000.csv",
        "api_data_aadhar_demographic_1500000_2000000.csv",
        "api_data_aadhar_demographic_500000_1000000.csv",
        "api_data_aadhar_demographic_2000000_2071700.csv",
        "api_data_aadhar_demographic_0_500000.csv"
    ]
    df_demo_list = []
    for f in demo_files:
        try:
            df_demo_list.append(pd.read_csv(f))
        except FileNotFoundError:
            continue
            
    if df_demo_list:
        df_demo = pd.concat(df_demo_list)
        df_demo.rename(columns={
            'demo_age_5_17': 'Demographic_5_17',
            'demo_age_17_': 'Demographic_18_plus'
        }, inplace=True)
    else:
        df_demo = pd.DataFrame(columns=['date', 'state', 'district', 'pincode', 'Demographic_5_17', 'Demographic_18_plus'])

    # 3. Load Enrolment Data (New Section)
    enrol_files = [
        "api_data_aadhar_enrolment_1000000_1006029.csv"
    ]
    df_enrol_list = []
    for f in enrol_files:
        try:
            df_enrol_list.append(pd.read_csv(f))
        except FileNotFoundError:
            continue
    
    if df_enrol_list:
        df_enrol = pd.concat(df_enrol_list)
        df_enrol.rename(columns={
            'age_0_5': 'Enrolment_0_5',
            'age_5_17': 'Enrolment_5_17',
            'age_18_greater': 'Enrolment_18_plus'
        }, inplace=True)
    else:
        df_enrol = pd.DataFrame(columns=['date', 'state', 'district', 'pincode', 'Enrolment_0_5', 'Enrolment_5_17', 'Enrolment_18_plus'])

    # 4. Merge Datasets
    # We group by common columns first to handle duplicates if any
    group_cols = ['date', 'state', 'district', 'pincode']
    
    df_bio_grouped = df_bio.groupby(group_cols, as_index=False).sum()
    df_demo_grouped = df_demo.groupby(group_cols, as_index=False).sum()
    df_enrol_grouped = df_enrol.groupby(group_cols, as_index=False).sum()

    # Outer join to keep all records: Start with Demo and Bio
    df_merged = pd.merge(df_demo_grouped, df_bio_grouped, on=group_cols, how='outer').fillna(0)
    
    # Merge Enrolment Data
    df_merged = pd.merge(df_merged, df_enrol_grouped, on=group_cols, how='outer').fillna(0)

    # Convert Date
    df_merged['date'] = pd.to_datetime(df_merged['date'], dayfirst=True, errors='coerce')
    
    # Calculate Totals
    df_merged['Total_Demographic_Updates'] = df_merged['Demographic_5_17'] + df_merged['Demographic_18_plus']
    df_merged['Total_Biometric_Updates'] = df_merged['Biometric_5_17'] + df_merged['Biometric_18_plus']
    df_merged['Total_Enrolments'] = df_merged['Enrolment_0_5'] + df_merged['Enrolment_5_17'] + df_merged['Enrolment_18_plus']
    
    # Clean string columns
    df_merged['state'] = df_merged['state'].astype(str).str.title().str.strip()
    df_merged['district'] = df_merged['district'].astype(str).str.title().str.strip()

    return df_merged

# --- Load Data ---
try:
    df = load_and_process_data()
except Exception as e:
    st.error(f"Error loading data. Please ensure CSV files are uploaded. Details: {e}")
    st.stop()

if df.empty:
    st.warning("No data found in the CSV files.")
    st.stop()

# --- Top Navigation Bar (Filters) ---
# Moved from sidebar to top columns to create a navbar feel
st.subheader("🔍 Filters Analysis")

# Create three columns for the filters
nav_col1, nav_col2, nav_col3 = st.columns(3)

# Date Filter
min_date = df['date'].min()
max_date = df['date'].max()

with nav_col1:
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

# State Filter
all_states = sorted(df['state'].unique())
with nav_col2:
    # Removed default selection to ensure no filters are applied initially
    selected_states = st.multiselect("Select State(s)", all_states)

# District Filter (Dynamic)
if selected_states:
    filtered_districts = sorted(df[df['state'].isin(selected_states)]['district'].unique())
else:
    filtered_districts = sorted(df['district'].unique())

with nav_col3:
    selected_districts = st.multiselect("Select District(s) (Optional)", filtered_districts)

# Add a separator line to distinguish the "navbar" from content
st.markdown("---")

# --- Filtering Logic ---
mask = (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
if selected_states:
    mask = mask & (df['state'].isin(selected_states))
if selected_districts:
    mask = mask & (df['district'].isin(selected_districts))

df_filtered = df.loc[mask]

# --- Main Dashboard ---
st.title("📍 Migration Hotspots & Service Demand Predictor")
st.markdown("Analyzing **Demographic vs. Biometric Updates** to detect migration patterns and predict public service needs.")

# 1. Top Level Metrics
col1, col2, col3, col4 = st.columns(4)
total_demo = df_filtered['Total_Demographic_Updates'].sum()
total_bio = df_filtered['Total_Biometric_Updates'].sum()
top_state = df_filtered.groupby('state')['Total_Demographic_Updates'].sum().idxmax() if not df_filtered.empty else "N/A"
top_district = df_filtered.groupby('district')['Total_Demographic_Updates'].sum().idxmax() if not df_filtered.empty else "N/A"

with col1:
    st.metric("Total Demographic Updates", f"{total_demo:,.0f}", delta="Migration Signal")
with col2:
    st.metric("Total Biometric Updates", f"{total_bio:,.0f}", delta="Routine Activity", delta_color="off")
with col3:
    st.metric("Highest Activity State", top_state)
with col4:
    st.metric("Top Hotspot District", top_district)

st.markdown("---")

# --- Tabs for Analysis ---
# Expanded tabs to include more detailed visualizations
# Added 'New Enrolment Trends' as Tab 7 to incorporate new data file analysis without disturbing others
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Migration Intensity Map", 
    "🔮 Service Demand Prediction", 
    "📈 Biometric vs Demographic",
    "📅 Temporal Trends", 
    "👥 Age Demographics",
    "📜 Strategic Policy Recommendations",
    "🆕 Enrolment Trends"
])

# --- TAB 1: Migration Intensity (Geographical) ---
with tab1:
    st.subheader("Geographical Distribution of Address Changes (Migration Proxy)")
    st.markdown("""
    This **Treemap** visualizes the intensity of Demographic Updates. 
    - **Size of Box**: Volume of Demographic Updates (Potential Migration Inflow/Correction).
    - **Color**: Ratio of Demographic to Biometric updates. Darker colors indicate areas where address changes significantly outweigh routine biometric updates.
    """)

    # Prepare Data for Treemap
    df_tree = df_filtered.groupby(['state', 'district']).agg({
        'Total_Demographic_Updates': 'sum',
        'Total_Biometric_Updates': 'sum'
    }).reset_index()
    
    # Calculate Ratio for Color Scale (Handling division by zero)
    df_tree['Update_Ratio'] = df_tree['Total_Demographic_Updates'] / (df_tree['Total_Biometric_Updates'] + 1)

    fig_tree = px.treemap(
        df_tree,
        path=[px.Constant("India"), 'state', 'district'],
        values='Total_Demographic_Updates',
        color='Total_Demographic_Updates', # Color by volume to match prompt "Top States lead volume"
        color_continuous_scale='RdYlBu_r',
        title="Demographic Update Intensity: State > District Hierarchy",
        hover_data=['Total_Biometric_Updates']
    )
    # Update layout for Navy theme
    fig_tree.update_layout(height=600, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_tree, use_container_width=True)

    # Detailed Explanation for Treemap
    st.markdown("""
    <div class="explanation-text">
    <b>🔍 Detailed Visualization Analysis:</b><br>
    The Treemap provides a hierarchical view of the country. The larger rectangles represent states with higher overall activity. 
    Inside each state, the districts are sized by their contribution to the migration volume. 
    <br><br>
    <b>🧠 Module Explanation:</b><br>
    Our system aggregates raw update counts. By using a heat-map color scale based on volume, we instantly highlight 'hotspots'. 
    A large dark red box indicates a district experiencing massive demographic shifts, likely due to an influx of workers or families updating addresses.
    <br><br>
    <b>🏛️ Government Action:</b><br>
    Districts that appear disproportionately large in this map should be prioritized for immediate infrastructure auditing. 
    State governments should deploy observers to these top 5 red zones to verify if the population surge is temporary (seasonal labor) or permanent.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Top 10 Districts Bar Chart
    st.subheader("🏆 Top 10 Destination Districts (by Demographic Volume)")
    top_10_districts = df_tree.sort_values(by='Total_Demographic_Updates', ascending=False).head(10)
    
    fig_bar = px.bar(
        top_10_districts,
        x='Total_Demographic_Updates',
        y='district',
        color='state',
        orientation='h',
        title="Districts with Highest Demographic Updates",
        labels={'Total_Demographic_Updates': 'Number of Updates', 'district': 'District'},
        text_auto='.2s'
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed Explanation for Bar Chart
    st.markdown("""
    <div class="explanation-text">
    <b>🔍 Detailed Visualization Analysis:</b><br>
    This horizontal bar chart strips away the geographical context to strictly rank districts by raw volume. 
    It forces a comparison across state lines, showing if a district in State A is facing higher pressure than a capital city in State B.
    <br><br>
    <b>🏛️ Government Action:</b><br>
    These top 10 districts account for the majority of administrative burden. 
    <b>Recommendation:</b> Open temporary Aadhaar Seva Kendras (ASKs) in these specific districts to reduce wait times and prevent backlog.
    </div>
    """, unsafe_allow_html=True)

# --- TAB 2: Service Demand Prediction ---
with tab2:
    st.subheader("🔮 Predictive Service Planning")
    st.markdown("""
    <div class="insight-box">
    <b>Logic:</b> 
    High demographic updates in specific age groups indicate new residency or data correction after migration.
    <br>• <b>Age 5-17 Updates</b> → Predicts demand for <b>Schools & Education facilities</b>.
    <br>• <b>Age 18+ Updates</b> → Predicts demand for <b>Ration Cards, Housing, and Jobs</b>.
    </div>
    """, unsafe_allow_html=True)

    col_edu, col_civic = st.columns(2)

    # Education Demand Analysis
    with col_edu:
        st.markdown("#### 🏫 School Capacity Planning")
        df_edu = df_filtered.groupby(['state', 'district'])['Demographic_5_17'].sum().reset_index()
        df_edu = df_edu.sort_values(by='Demographic_5_17', ascending=False).head(10)
        
        fig_edu = px.bar(
            df_edu, x='district', y='Demographic_5_17', color='state',
            title="Top Districts: Potential School Demand (Age 5-17)",
            labels={'Demographic_5_17': 'Student Age Updates'}
        )
        fig_edu.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_edu, use_container_width=True)
        
        st.info(f"Insight: {df_edu.iloc[0]['district']} ({df_edu.iloc[0]['state']}) shows the highest flux in student-age population.")

        st.markdown("""
        <div class="explanation-text">
        <b>🧠 Module Logic:</b> We filter updates specifically for the 5-17 age bracket. An address update here strongly correlates with school transfers.
        <br>
        <b>🏛️ Action:</b> The Education Ministry must alert the District Education Officer (DEO) in these top districts to check school enrollment capacities for the upcoming academic year.
        </div>
        """, unsafe_allow_html=True)

    # Civic Services Demand Analysis
    with col_civic:
        st.markdown("#### 🏠 Housing & Ration Card Planning")
        df_civic = df_filtered.groupby(['state', 'district'])['Demographic_18_plus'].sum().reset_index()
        df_civic = df_civic.sort_values(by='Demographic_18_plus', ascending=False).head(10)
        
        fig_civic = px.bar(
            df_civic, x='district', y='Demographic_18_plus', color='state',
            title="Top Districts: Potential Ration/Housing Demand (Age 18+)",
            labels={'Demographic_18_plus': 'Adult Updates'}
        )
        fig_civic.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_civic, use_container_width=True)

        st.info(f"Insight: {df_civic.iloc[0]['district']} ({df_civic.iloc[0]['state']}) requires immediate review of public distribution systems (PDS).")

        st.markdown("""
        <div class="explanation-text">
        <b>🧠 Module Logic:</b> Updates in the 18+ category often precede applications for local voting rights, ration cards, and housing benefits.
        <br>
        <b>🏛️ Action:</b> Municipal corporations should prepare for a surge in civic amenity usage. PDS (Ration) shops in these zones should be stocked with additional grain quotas.
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: Biometric vs Demographic Correlation ---
with tab3:
    st.subheader("Type of Activity Analysis")
    st.markdown("Distinguishing between **Routine Maintenance** (Linear correlation) and **Migration Events** (Outliers).")
    
    # Aggregating by District
    df_scatter = df_filtered.groupby(['state', 'district']).agg({
        'Total_Demographic_Updates': 'sum',
        'Total_Biometric_Updates': 'sum'
    }).reset_index()

    fig_scatter = px.scatter(
        df_scatter,
        x='Total_Biometric_Updates',
        y='Total_Demographic_Updates',
        color='state',
        size='Total_Demographic_Updates',
        hover_name='district',
        title="Biometric (Routine) vs. Demographic (Migration) Updates",
        labels={
            'Total_Biometric_Updates': 'Biometric Updates (Routine)',
            'Total_Demographic_Updates': 'Demographic Updates (Address Changes)'
        }
    )
    
    # Add a reference line
    fig_scatter.add_shape(type="line",
        x0=0, y0=0, x1=df_scatter['Total_Biometric_Updates'].max(), y1=df_scatter['Total_Demographic_Updates'].max(),
        line=dict(color="Gray", width=1, dash="dash")
    )

    fig_scatter.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("""
    <div class="explanation-text">
    <b>🔍 Detailed Visualization Analysis:</b><br>
    This scatter plot places every district on a grid of Biometric vs Demographic activity. 
    The <b>Dashed Line</b> represents a 1:1 ratio, which is the baseline for normal administrative activity.
    <br><br>
    <b>🧠 Module Explanation (Outliers):</b><br>
    - <b>Points far ABOVE the line:</b> These are "Migration Hotspots". The ratio of address changes is abnormally high compared to routine biometric updates (like age 5/15 updates). This signals new people arriving.
    - <b>Points far BELOW the line:</b> These are "Stable Districts". High biometric activity (likely children turning 5 or 15) but very few address changes.
    <br><br>
    <b>🏛️ Government Action:</b><br>
    Focus policy interventions <i>only</i> on the districts above the line. Districts below the line are functioning normally and require no special intervention.
    </div>
    """, unsafe_allow_html=True)

# --- TAB 4: Temporal Trends ---
with tab4:
    st.subheader("📅 Temporal Trends: When is Migration Happening?")
    st.markdown("Analyzing the timeline of updates to identify seasonal patterns or event-triggered migration.")

    # Group by Date
    df_time = df_filtered.groupby('date')[['Total_Demographic_Updates', 'Total_Biometric_Updates']].sum().reset_index()

    fig_line = px.line(
        df_time, 
        x='date', 
        y=['Total_Demographic_Updates', 'Total_Biometric_Updates'],
        title="Timeline of Updates: Biometric vs Demographic",
        labels={'value': 'Number of Updates', 'date': 'Date', 'variable': 'Update Type'},
        markers=True
    )
    fig_line.update_layout(hovermode="x unified", template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("""
    <div class="explanation-text">
    <b>🔍 Detailed Visualization Analysis:</b><br>
    This line chart tracks the daily volume of updates. The two lines allow us to compare the *volatility* of migration (Demographic) versus the *stability* of routine work (Biometric).
    <br><br>
    <b>🧠 Module Explanation:</b><br>
    If the blue line (Demographic) spikes suddenly while the red line (Biometric) remains flat, it indicates a "Migration Shock Event"—such as a festival end, harvest season end, or a natural disaster forcing relocation.
    <br><br>
    <b>🏛️ Government Action:</b><br>
    If a recurring seasonal spike is observed (e.g., every June), the government should pre-allocate resources (train tickets, temporary shelters) in source and destination districts *before* the spike occurs next year.
    </div>
    """, unsafe_allow_html=True)

# --- TAB 5: Age Demographics ---
with tab5:
    st.subheader("👥 Age Group Composition: Who is Moving?")
    st.markdown("Analyzing the demographic split to understand the nature of the population shift.")

    col_age1, col_age2 = st.columns([2, 1])

    with col_age1:
        # Stacked Bar Chart for Age Groups by State
        df_age_state = df_filtered.groupby('state')[['Demographic_5_17', 'Demographic_18_plus']].sum().reset_index()
        
        fig_age_stack = px.bar(
            df_age_state, 
            x='state', 
            y=['Demographic_5_17', 'Demographic_18_plus'],
            title="Demographic Composition by State (Stacked)",
            labels={'value': 'Count', 'variable': 'Age Group'},
            barmode='stack'
        )
        fig_age_stack.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_age_stack, use_container_width=True)

    with col_age2:
        # Pie Chart for Overall Mix
        total_5_17 = df_filtered['Demographic_5_17'].sum()
        total_18_plus = df_filtered['Demographic_18_plus'].sum()
        
        fig_pie = px.pie(
            names=['Age 5-17 (Minors)', 'Age 18+ (Adults)'],
            values=[total_5_17, total_18_plus],
            title="Overall Migrant Age Split",
            hole=0.4
        )
        fig_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("""
    <div class="explanation-text">
    <b>🔍 Detailed Visualization Analysis:</b><br>
    The stacked bar chart breaks down the total volume into two critical segments: Minors and Adults. The Pie chart gives a national-level summary.
    <br><br>
    <b>🧠 Module Explanation:</b><br>
    - <b>High Blue Portion (5-17):</b> Indicates family migration. Workers are moving <i>with</i> their children. This puts pressure on schools and healthcare (vaccinations).
    - <b>High Red Portion (18+):</b> Indicates economic migration. Likely single male/female workers moving for jobs, leaving families behind. This puts pressure on housing and transport.
    <br><br>
    <b>🏛️ Government Action:</b><br>
    States with high "Blue" bars must focus on <b>Anganwadis and Schools</b>.
    States with high "Red" bars must focus on <b>Rental Housing and Labor Regulations</b>.
    </div>
    """, unsafe_allow_html=True)

# --- TAB 6: Strategic Policy Recommendations ---
with tab6:
    st.subheader("📜 Strategic Policy Recommendations")
    st.markdown("Based on the comprehensive analysis of biometric and demographic data, the following strategic actions are recommended for the government.")

    st.info("These recommendations are generated dynamically based on the patterns observed in the uploaded datasets.")

    col_rec1, col_rec2 = st.columns(2)

    with col_rec1:
        st.markdown("""
        <div class="metric-card">
        <h3 class="recommendation-header">🏗️ Infrastructure & Planning</h3>
        <ul style="text-align: left; margin-top: 10px;">
            <li><b>Targeted Resource Allocation:</b> Stop uniform budget distribution. Divert extra administrative funds specifically to the Top 10 districts identified in Tab 1.</li>
            <li><b>School Expansion:</b> In districts where the 5-17 demographic update ratio exceeds 30%, authorize immediate hiring of contract teachers and construction of temporary classrooms.</li>
            <li><b>Urban Planning:</b> Update city master plans in 'Hotspot' districts to account for higher population density, focusing on water supply and waste management.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_rec2:
        st.markdown("""
        <div class="metric-card">
        <h3 class="recommendation-header">⚖️ Social Welfare & Security</h3>
        <ul style="text-align: left; margin-top: 10px;">
            <li><b>PDS Portability (One Nation One Ration):</b> Ensure 100% implementation of ration card portability in districts with high 18+ demographic inflow (Tab 2).</li>
            <li><b>Labor Rights:</b> Conduct special labor registration drives in districts identified as "Migration Hotspots" in Tab 3 to ensure workers are not exploited.</li>
            <li><b>Health Surveillance:</b> Mobile populations are vectors for disease. deploy extra mobile health units (Mohalla Clinics) in the high-inflow zones.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 7: Enrolment Trends (NEW) ---
with tab7:
    st.subheader("🆕 New Enrolment Analysis")
    st.markdown("Tracking **Fresh Aadhaar Generations** to understand natural population growth vs. delayed registrations.")
    
    # Check if we have enrolment data
    if 'Total_Enrolments' in df_filtered.columns and df_filtered['Total_Enrolments'].sum() > 0:
        col_enrol1, col_enrol2 = st.columns(2)
        
        with col_enrol1:
            total_new_enrolments = df_filtered['Total_Enrolments'].sum()
            st.metric("Total New Enrolments Generated", f"{total_new_enrolments:,.0f}", delta="Population Growth")
            
            # Age split for enrolments
            enrol_0_5 = df_filtered['Enrolment_0_5'].sum()
            enrol_5_17 = df_filtered['Enrolment_5_17'].sum()
            enrol_18_plus = df_filtered['Enrolment_18_plus'].sum()
            
            fig_enrol_pie = px.pie(
                names=['Age 0-5 (Infants)', 'Age 5-17 (Minors)', 'Age 18+ (Adults)'],
                values=[enrol_0_5, enrol_5_17, enrol_18_plus],
                title="Age Composition of New Enrolments",
                hole=0.4
            )
            fig_enrol_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_enrol_pie, use_container_width=True)

        with col_enrol2:
            st.markdown("#### Top Districts for New Registrations")
            df_enrol_dist = df_filtered.groupby(['state', 'district'])['Total_Enrolments'].sum().reset_index()
            df_enrol_dist = df_enrol_dist.sort_values(by='Total_Enrolments', ascending=False).head(10)
            
            fig_enrol_bar = px.bar(
                df_enrol_dist,
                x='Total_Enrolments',
                y='district',
                color='state',
                orientation='h',
                title="Districts with Highest New Enrolments",
                labels={'Total_Enrolments': 'New Aadhaar Enrolments'}
            )
            fig_enrol_bar.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_enrol_bar, use_container_width=True)

        st.markdown("""
        <div class="explanation-text">
        <b>🔍 Enrolment Insight:</b><br>
        This tab visualizes *new* identities being created. 
        <br>
        - <b>High 0-5 Age Group:</b> Indicates healthy birth registration rates in the region.
        - <b>High 18+ Age Group:</b> Indicates a backlog of unregistered adults, often in remote or tribal areas, now entering the formal economy.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No Enrolment Data available in the current filtered selection.")

st.markdown("---")
st.markdown("### 🔚 Conclusion")
st.markdown("""
This dashboard serves as a **Decision Support System (DSS)**. By moving from reactive data viewing to proactive predictive modeling, the administration can shift from "managing crises" to "managing growth".
The distinction between **Biometric (Routine)** and **Demographic (Migration)** updates is the key key variable that unlocks this predictive capability.
""")

# --- Footer ---
st.markdown("---")
# st.caption("Data Source: Uploaded Aadhar API Datasets (Biometric, Demographic, Enrolment). Analysis generated by AI.")