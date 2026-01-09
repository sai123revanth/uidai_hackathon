import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
from groq import Groq  # Import Groq Client

# --- 1. SEO & PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Demographic Dividend | Education vs Workforce",
    page_icon="🤖",  # Updated to Chatbot Logo
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
    
    h1, h2, h3, h4 { 
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

    /* --- FLOATING CHATBOT BUTTON (MODERN FAB) --- */
    /* Container positioning */
    div[data-testid="stPopover"] {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
        width: auto;
    }
    
    /* Circular Button Styling */
    div[data-testid="stPopover"] > button {
        width: 60px !important;
        height: 60px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #00CC96 0%, #00a87d 100%) !important;
        box-shadow: 0 8px 32px rgba(0, 204, 150, 0.4) !important;
        border: 2px solid rgba(255,255,255,0.1) !important;
        color: white !important;
        font-size: 28px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    /* Hover Effect */
    div[data-testid="stPopover"] > button:hover {
        transform: translateY(-5px) scale(1.05) !important;
        box-shadow: 0 12px 40px rgba(0, 204, 150, 0.6) !important;
        background: linear-gradient(135deg, #00e0a6 0%, #00bc8b 100%) !important;
    }
    
    div[data-testid="stPopover"] > button:active {
        transform: scale(0.95) !important;
    }
    
    /* Popover Body Styling */
    div[data-testid="stPopoverBody"] {
        border-radius: 12px;
        border: 1px solid #334155;
        background-color: #0f172a; /* Dark background matches theme */
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        padding: 0 !important; /* Remove default padding for full control */
    }

    /* --- MOBILE OPTIMIZATION --- */
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
        .js-plotly-plot {
            height: 400px !important;
        }
        div[data-testid="stPopover"] {
            bottom: 20px;
            right: 20px;
        }
        div[data-testid="stPopover"] > button {
            width: 50px !important;
            height: 50px !important;
            font-size: 24px !important;
        }
    }
</style>
"""
st.markdown(seo_meta_tags, unsafe_allow_html=True)

# --- 3. GROQ AI CLIENT SETUP ---
def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except KeyError:
        st.error("🚨 GROQ_API_KEY not found in secrets.toml. AI features will be disabled.")
        return None

client = get_groq_client()

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    file_pattern = "api_data_aadhar_demographic_*.csv"
    files = glob.glob(file_pattern)
    
    if not files:
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
    
    return raw_df

# Load raw Data
raw_df_full = load_data()

if raw_df_full is None:
    st.error("No data found. Please place the CSV files in the same directory.")
    st.stop()

# --- 4. DEEP LINKING SETUP ---
query_params = st.query_params
default_states = query_params.get_all("state") if "state" in query_params else []
valid_states = sorted(raw_df_full['State'].unique())
default_states = [s for s in default_states if s in valid_states]

# --- MAIN DASHBOARD HEADER ---
col_head_1, col_head_2 = st.columns([4, 1])
with col_head_1:
    st.title("🤖 The Demographic Dividend: Education vs. Workforce Intelligence")

# DETAILED INTRODUCTION
st.markdown("""
<div style='background-color: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 10px; border-left: 5px solid #00CC96; margin-bottom: 25px;'>
    <h4 style='margin-top:0;'>📌 Module Objective</h4>
    <p style='font-size: 1.05em; color: #e2e8f0;'>
        This module transforms raw Aadhar update logs into a <b>Policy Strategic Tool</b>. 
        Because demographic updates (like address or biometric changes) often coincide with milestones like school enrollment or opening a bank account, 
        we use these signals to map India's "Age Profile" by district.
    </p>
    <div style='display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px;'>
        <div style='flex: 1; min-width: 250px;'>
            <b style='color:#00CC96;'>🟢 HIGH YOUTH INDEX:</b> Districts with a high ratio of updates from <b>Age 5-17</b>. 
            <i>Policy Goal: Scholarships, Digital Classrooms, Primary Health.</i>
        </div>
        <div style='flex: 1; min-width: 250px;'>
            <b style='color:#EF553B;'>🔴 LOW YOUTH INDEX:</b> Districts dominated by <b>Age 17+</b> updates. 
            <i>Policy Goal: Banking Expansion, Skill India Centers, Micro-Credit.</i>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- FILTER BAR ---
st.markdown("### 🔍 Dashboard Filters")
st.caption("Customize the view to focus on specific states, timelines, or remove statistical noise.")
col_filter_1, col_filter_2, col_filter_3 = st.columns([1, 1, 1])

with col_filter_1:
    selected_states = st.multiselect(
        "Focus on Specific States",
        options=valid_states,
        default=default_states,
        help="Analyzing specific states allows for regional benchmarking."
    )

with col_filter_2:
    # Timeline Selection
    min_date = raw_df_full['Date'].min().date()
    max_date = raw_df_full['Date'].max().date()
    
    selected_date_range = st.date_input(
        "Select Timeline",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter data by a specific period to see demographic shifts over time."
    )

with col_filter_3:
    min_updates = st.slider(
        "Filter Noise (Minimum Volume)",
        min_value=0,
        max_value=int(raw_df_full.groupby(['State', 'District'])['Youth_Updates'].sum().quantile(0.9)), 
        value=100,
        step=50,
        help="We recommend a minimum of 100 for statistical accuracy."
    )

# --- DATA FILTERING & AGGREGATION ---
# 1. Timeline Filter
if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_raw = raw_df_full[
        (raw_df_full['Date'].dt.date >= start_date) & 
        (raw_df_full['Date'].dt.date <= end_date)
    ]
else:
    filtered_raw = raw_df_full

# 2. State Filter Logic & Query Param Update
if selected_states:
    st.query_params["state"] = selected_states
    filtered_raw = filtered_raw[filtered_raw['State'].isin(selected_states)]
else:
    if "state" in st.query_params:
        del st.query_params["state"]

# 3. Aggregations based on Timeline/State
district_df = filtered_raw.groupby(['State', 'District'])[['Youth_Updates', 'Adult_Updates']].sum().reset_index()
district_df['Total_Updates'] = district_df['Youth_Updates'] + district_df['Adult_Updates']
district_df['Youth_Index'] = (district_df['Youth_Updates'] / district_df['Total_Updates']) * 100

filtered_raw['Month_Year'] = filtered_raw['Date'].dt.to_period('M').astype(str)
trend_df = filtered_raw.groupby(['Month_Year'])[['Youth_Updates', 'Adult_Updates']].sum().reset_index()
trend_df['Total_Updates'] = trend_df['Youth_Updates'] + trend_df['Adult_Updates']

# 4. Noise Filter
filtered_df = district_df[district_df['Total_Updates'] >= min_updates]

# ==========================================
# 🤖 AI CHATBOT LOGIC HELPERS
# ==========================================
def prepare_data_context(df, state_selection):
    """Summarizes current data view for the AI Context"""
    if df.empty:
        return "No data available for the current selection."
    
    total_samples = df['Total_Updates'].sum()
    avg_youth_idx = df['Youth_Index'].mean()
    
    top_youth = df.nlargest(5, 'Youth_Index')[['District', 'State', 'Youth_Index']]
    top_work = df.nsmallest(5, 'Youth_Index')[['District', 'State', 'Youth_Index']]
    
    context = f"""
    DATA CONTEXT:
    - Scope: {state_selection if state_selection else "All India"}
    - Total Data Points: {total_samples}
    - Average Youth Index: {avg_youth_idx:.2f}%
    
    TOP 5 DISTRICTS NEEDING EDUCATION FOCUS (High Youth Index):
    {top_youth.to_string(index=False)}
    
    TOP 5 DISTRICTS NEEDING JOB FOCUS (Low Youth Index/High Adult):
    {top_work.to_string(index=False)}
    """
    return context

def get_ai_response(messages):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- TABS ---
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
    st.markdown("#### ⚡ Real-Time Snapshot")
    st.caption("These metrics provide an immediate sense of the scale and demographic lean of your current selection.")
    
    total_vol = filtered_df['Total_Updates'].sum()
    avg_index = filtered_df['Youth_Index'].mean()
    
    if not filtered_df.empty:
        youngest = filtered_df.loc[filtered_df['Youth_Index'].idxmax()]
        maturest = filtered_df.loc[filtered_df['Youth_Index'].idxmin()]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sample Size", f"{total_vol:,.0f}", help="Total updates analyzed across all age groups.")
        c2.metric("Avg Youth Density", f"{avg_index:.1f}%", help="The average percentage of children (5-17) in this region.")
        c3.metric("Youngest Dist.", f"{youngest['District']}", f"{youngest['Youth_Index']:.1f}% Youth") 
        c4.metric("Maturest Dist.", f"{maturest['District']}", f"{100-maturest['Youth_Index']:.1f}% Workforce") 
    else:
        st.warning("No data meets the current filter criteria. Try adjusting the 'Timeline' or 'Noise Filter'.")

    st.markdown("---")
    
    st.subheader("📍 The National Demographic Heatmap")
    st.markdown("**How to read this chart:**")
    st.markdown("""
    - **Box Size:** Larger boxes mean more people are updating their records (High Activity).
    - **Box Color:** **Yellow/Light Green** = High Children ratio. **Dark Blue/Purple** = High Adult ratio.
    - **Click a State:** You can click on a state to zoom into its specific districts.
    """)
    
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
    
    st.markdown("---")
    st.markdown("### 🏛️ Targeted Policy Action Plan")
    st.caption("Automatically generated priority lists based on the 'Youth Index'.")
    
    ac1, ac2 = st.columns(2)
    cols_to_show = ['State', 'District', 'Youth_Index', 'Total_Updates']
    
    with ac1:
        st.info("🎒 **Top Education Priority (Highest Youth Index)**")
        st.markdown("*Recommended Action:* Prioritize New Primary Schools and Scholarship disbursement.")
        top_youth = filtered_df.nlargest(10, 'Youth_Index')
        st.dataframe(
            top_youth[cols_to_show].style.format({"Youth_Index": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

    with ac2:
        st.error("💼 **Top Workforce Priority (Lowest Youth Index)**")
        st.markdown("*Recommended Action:* Focus on Vocational Training, Job Fairs, and Bank credit access.")
        top_work = filtered_df.nsmallest(10, 'Youth_Index').copy()
        top_work['Adult_Index'] = 100 - top_work['Youth_Index']
        st.dataframe(
            top_work[['State', 'District', 'Adult_Index', 'Total_Updates']].style.format({"Adult_Index": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# TAB 2: STATE ANALYTICS
# ==========================================
with tab2:
    st.subheader("🏢 State-Level Comparative Benchmarking")
    st.markdown("""
    This tab compares states as a whole. This is crucial for **Federal Fund Allocation**. 
    States on the left of the chart are "Younger" and may require more Education Budget.
    States on the right are "Older" and may require more Employment/Industrial support.
    """)
    
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
            title="State Age Profiles (Youngest to Oldest)",
            labels={'Youth_Index': 'Avg Youth Index (%)'},
            height=500
        )
        fig_state.add_hline(y=state_stats['Youth_Index'].mean(), line_dash="dot", annotation_text="National Avg")
        st.plotly_chart(fig_state, use_container_width=True)
        
    with col_b:
        st.write("#### State Performance Stats")
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
                    "Update Volume",
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
    st.subheader("📅 Temporal Demographic Momentum")
    st.markdown("""
    **The Insight:** This chart shows how the demographic profile shifts over the months. 
    A sudden spike in **Youth Updates** (Green) might indicate a school enrollment season, while a rise in **Adult Updates** (Red) 
    could signal a migration for work or a new welfare scheme launch.
    """)
    
    if trend_df is not None and not trend_df.empty:
        trend_df = trend_df.sort_values('Month_Year')
        
        fig_trend = px.area(
            trend_df, 
            x='Month_Year', 
            y=['Youth_Updates', 'Adult_Updates'],
            title="Update Volume Trends (Selected Period): Youth vs Workforce Segment",
            labels={'value': 'Monthly Updates', 'variable': 'Demographic Segment'},
            color_discrete_map={'Youth_Updates': '#00CC96', 'Adult_Updates': '#EF553B'}
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Not enough temporal data available to show trends.")

# ==========================================
# TAB 4: DISTRICT EXPLORER
# ==========================================
with tab4:
    st.subheader("🔎 Granular District Intelligence")
    st.markdown("Use this tab to search for specific districts and see exactly where they sit in the national ecosystem.")
    
    c_search, c_chart = st.columns([1, 2])
    
    with c_search:
        st.markdown("#### Search & Distribution")
        search_term = st.text_input("Enter District Name", placeholder="e.g., Mahabubnagar")
        
        display_df = filtered_df.copy()
        if search_term:
            display_df = display_df[display_df['District'].str.contains(search_term, case=False)]
            
        fig_hist = px.histogram(
            filtered_df, 
            x="Youth_Index", 
            nbins=30, 
            title="Statistical Bell Curve of Districts",
            color_discrete_sequence=['#636EFA']
        )
        fig_hist.update_layout(showlegend=False, height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Shows how many districts fall into which 'Youth Index' percentage.")

    with c_chart:
        fig_scatter = px.scatter(
            display_df,
            x='Total_Updates',
            y='Youth_Index',
            size='Total_Updates',
            color='State',
            hover_name='District',
            log_x=True,
            title="Strategic Quadrants (Volume vs Demographic Profile)",
            height=500
        )
        fig_scatter.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5, annotation_text="Balance Line")
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("**X-Axis (Log Scale):** Shows the activity volume. **Y-Axis:** Shows the Youth vs Workforce balance.")

    st.markdown("### 📋 Full Analytic Table")
    st.caption("Scroll or search to see the full data for all districts.")
    
    mobile_friendly_cols = ['District', 'State', 'Total_Updates', 'Youth_Index']
    
    st.dataframe(
        display_df[mobile_friendly_cols].sort_values(by='Total_Updates', ascending=False),
        column_config={
            "Youth_Index": st.column_config.ProgressColumn(
                "Youth Index (%)",
                help="Higher = Needs Schools. Lower = Needs Jobs.",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Total_Updates": st.column_config.NumberColumn(
                "Activity Volume",
                format="%d"
            )
        },
        use_container_width=True,
        hide_index=True
    )

# --- RENDER FLOATING CHATBOT BUTTON (FAB) ---
if client:
    # Use st.popover to create a floating chat window
    # The CSS defined at the top will position this element at the bottom right
    with st.popover("💬", use_container_width=False):
        
        # --- FIXED HEADER (Sticky) ---
        st.markdown(
            """
            <div style="
                position: sticky; 
                top: 0; 
                background-color: #0f172a; 
                z-index: 1000; 
                padding: 15px 10px; 
                border-bottom: 1px solid #334155;
                margin-top: -1rem; /* Negate default padding */
                margin-left: -1rem;
                margin-right: -1rem;
            ">
                <div style="display: flex; align-items: center; justify-content: space-between; padding-left: 10px; padding-right: 10px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 24px;">🤖</span>
                        <div>
                            <h3 style="margin: 0; font-size: 16px; color: #f1f5f9;">Policy Advisor</h3>
                            <p style="margin: 0; font-size: 11px; color: #94a3b8;">AI Analytics Engine</p>
                        </div>
                    </div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Initialize session history for chat if not present
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Prepare context based on CURRENT filters passed to modal
        data_context = prepare_data_context(filtered_df, selected_states)
        
        # System Prompt
        system_prompt = {
            "role": "system", 
            "content": f"""You are a senior Indian Government Policy Advisor. 
            You are analyzing Aadhar demographic update data to suggest improvements.
            
            Metrics Definition:
            - Youth Index: High % means more children (5-17). Needs Schools, Scholarships, Nutrition.
            - Low Youth Index (High Adult): Means more workforce (17+). Needs Jobs, Skilling, Banking, Loans.
            
            Current Data:
            {data_context}
            
            Instructions:
            1. Provide specific, actionable policy advice based on the districts shown.
            2. If the user speaks in an Indian language (Hindi, Tamil, Telugu, Marathi, Bengali, etc.), REPLY IN THAT LANGUAGE.
            3. Keep answers concise and professional.
            """
        }

        # Auto-trigger specific insights if history is empty
        if not st.session_state.chat_history:
            initial_prompt = "Based on the currently filtered data, give me 3 urgent government policy recommendations to improve the situation."
            st.session_state.chat_history.append({"role": "user", "content": initial_prompt})
            
            # Generate response
            full_messages = [system_prompt] + st.session_state.chat_history
            with st.spinner("Analyzing demographic data for insights..."):
                response = get_ai_response(full_messages)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        # FIXED SIZE SCROLLABLE CONTAINER FOR CHAT HISTORY
        # This creates a scrollable area of fixed height (400px) inside the popover
        chat_container = st.container(height=400)
        with chat_container:
            # Display Chat History
            for message in st.session_state.chat_history:
                if message["role"] != "system": # Skip system prompt
                    if message["content"] == "Based on the currently filtered data, give me 3 urgent government policy recommendations to improve the situation.":
                            with st.chat_message("user"):
                                st.write("🔍 *Auto-Analysis Request*")
                    else:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

        # Chat Input inside Popover
        if prompt := st.chat_input("Ask policy questions...", key="popover_chat_input"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            full_messages = [system_prompt] + st.session_state.chat_history
            
            response = get_ai_response(full_messages)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()