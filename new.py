import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# --- 1. SEO & PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Demographic Dividend | Education vs Workforce",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.uidai.gov.in/',
        'Report a bug': "mailto:admin@example.com",
        'About': "# Demographic Dividend Dashboard\nThis tool visualizes Aadhar update trends."
    }
)

# --- 2. META TAGS & RESPONSIVE STYLING ---
# Includes custom CSS for the floating chat button and general aesthetics
seo_meta_tags = """
<div style="display: none;">
    <h1>India Demographic Dividend Dashboard</h1>
    <p>Analyze district-level demographic data.</p>
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

    /* AI CHAT BUTTON STYLING */
    div[data-testid="stPopover"] > button {
        background: linear-gradient(45deg, #00CC96, #38bdf8);
        color: white;
        border: none;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 204, 150, 0.3);
        transition: all 0.3s ease;
    }
    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 204, 150, 0.5);
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
    # Attempt to find columns dynamically or default
    youth_col = 'demo_age_5_17' if 'demo_age_5_17' in cols else cols[2] # Fallback logic
    adult_col = [c for c in cols if '18' in c or 'age_17' in c]
    adult_col = adult_col[0] if adult_col else cols[3]
    
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

# --- 3. FILTER LOGIC (Global) ---
query_params = st.query_params
default_states = query_params.get_all("state") if "state" in query_params else []
valid_states = sorted(raw_df_full['State'].unique())
default_states = [s for s in default_states if s in valid_states]

# --- 4. TOP HEADER LAYOUT WITH CHATBOT ---
col_title, col_chat = st.columns([5, 1], gap="small")

with col_title:
    st.title("🇮🇳 Demographic Dividend Dashboard")

# --- 🤖 MODERN AI CHATBOT INTEGRATION ---
with col_chat:
    # This creates the button in the top right
    with st.popover("🤖 AI Insight", use_container_width=True):
        st.markdown("### 🧠 Data Copilot")
        st.caption("Powered by Groq Llama-3.3")
        
        # 1. API Key Handling
        if "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        else:
            groq_api_key = st.text_input("Groq API Key", type="password")

        if not groq_api_key:
            st.warning("Enter Groq Key to chat.")
        else:
            # 2. Chat History Init
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "I analyzed the dashboard. Ask me for policy improvements or deep insights!"}
                ]

            # 3. Display Chat
            chat_container = st.container(height=300)
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            # 4. Input & Logic
            if prompt := st.chat_input("Ask about trends, improvements...", key="chat_input"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            try:
                                # Initialize LLM
                                llm = ChatGroq(
                                    temperature=0, 
                                    model_name="llama-3.3-70b-versatile", 
                                    api_key=groq_api_key
                                )
                                
                                # Prepare data for AI (Use filtered data logic from below, but simplified for AI context)
                                # We pass the FULL dataset to the AI so it can do its own filtering if asked, 
                                # but we tell it about the context in the prompt.
                                agent = create_pandas_dataframe_agent(
                                    llm,
                                    raw_df_full,
                                    verbose=True,
                                    allow_dangerous_code=True,
                                    handle_parsing_errors=True,
                                    max_iterations=5
                                )
                                
                                # Enhancing the prompt for "Improvements" and "Insights"
                                system_context = (
                                    f"User Query: {prompt}\n"
                                    "You are a Senior Policy Analyst. Use the dataframe. "
                                    "If asked for improvements, look for districts with low Youth Index (need jobs) or high Youth Index (need schools). "
                                    "Be specific, suggest concrete actions, and cite data numbers."
                                )
                                
                                response = agent.run(system_context)
                                st.write(response)
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            except Exception as e:
                                st.error(f"Analysis failed: {e}")

# --- DASHBOARD CONTENT CONTINUES ---

# Introduction
st.markdown("""
<div style='background-color: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 10px; border-left: 5px solid #00CC96; margin-bottom: 25px;'>
    <h4 style='margin-top:0;'>📌 Module Objective</h4>
    <p style='font-size: 1.05em; color: #e2e8f0;'>
        This module transforms raw Aadhar update logs into a <b>Policy Strategic Tool</b>. 
        High Youth Index = Education Priority. Low Youth Index = Workforce Priority.
    </p>
</div>
""", unsafe_allow_html=True)

# --- FILTER BAR ---
st.markdown("### 🔍 Dashboard Filters")
col_filter_1, col_filter_2, col_filter_3 = st.columns([1, 1, 1])

with col_filter_1:
    selected_states = st.multiselect(
        "Focus on Specific States",
        options=valid_states,
        default=default_states
    )

with col_filter_2:
    min_date = raw_df_full['Date'].min().date()
    max_date = raw_df_full['Date'].max().date()
    selected_date_range = st.date_input("Select Timeline", value=(min_date, max_date), min_value=min_date, max_value=max_date)

with col_filter_3:
    min_updates = st.slider("Filter Noise (Min Volume)", 0, 1000, 100, 50)

# --- DATA FILTERING & AGGREGATION ---
if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_raw = raw_df_full[(raw_df_full['Date'].dt.date >= start_date) & (raw_df_full['Date'].dt.date <= end_date)]
else:
    filtered_raw = raw_df_full

if selected_states:
    st.query_params["state"] = selected_states
    filtered_raw = filtered_raw[filtered_raw['State'].isin(selected_states)]
else:
    if "state" in st.query_params:
        del st.query_params["state"]

# Aggregations
district_df = filtered_raw.groupby(['State', 'District'])[['Youth_Updates', 'Adult_Updates']].sum().reset_index()
district_df['Total_Updates'] = district_df['Youth_Updates'] + district_df['Adult_Updates']
district_df['Youth_Index'] = (district_df['Youth_Updates'] / district_df['Total_Updates']) * 100

filtered_raw['Month_Year'] = filtered_raw['Date'].dt.to_period('M').astype(str)
trend_df = filtered_raw.groupby(['Month_Year'])[['Youth_Updates', 'Adult_Updates']].sum().reset_index()
trend_df['Total_Updates'] = trend_df['Youth_Updates'] + trend_df['Adult_Updates']

# Noise Filter
filtered_df = district_df[district_df['Total_Updates'] >= min_updates]

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
    total_vol = filtered_df['Total_Updates'].sum()
    avg_index = filtered_df['Youth_Index'].mean()
    
    if not filtered_df.empty:
        youngest = filtered_df.loc[filtered_df['Youth_Index'].idxmax()]
        maturest = filtered_df.loc[filtered_df['Youth_Index'].idxmin()]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Sample Size", f"{total_vol:,.0f}")
        c2.metric("Avg Youth Density", f"{avg_index:.1f}%")
        c3.metric("Youngest Dist.", f"{youngest['District']}", f"{youngest['Youth_Index']:.1f}% Youth") 
        c4.metric("Maturest Dist.", f"{maturest['District']}", f"{100-maturest['Youth_Index']:.1f}% Workforce") 
    else:
        st.warning("No data meets the current filter criteria.")

    st.markdown("---")
    st.subheader("📍 National Heatmap")
    
    fig_tree = px.treemap(
        filtered_df,
        path=[px.Constant("India"), 'State', 'District'],
        values='Total_Updates',
        color='Youth_Index',
        color_continuous_scale='Viridis',
        height=500
    )
    st.plotly_chart(fig_tree, use_container_width=True)
    
    st.markdown("---")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.info("🎒 **Top Education Priority**")
        top_youth = filtered_df.nlargest(10, 'Youth_Index')
        st.dataframe(top_youth[['State', 'District', 'Youth_Index']].style.format({"Youth_Index": "{:.1f}%"}), use_container_width=True, hide_index=True)
    with ac2:
        st.error("💼 **Top Workforce Priority**")
        top_work = filtered_df.nsmallest(10, 'Youth_Index')
        st.dataframe(top_work[['State', 'District', 'Youth_Index']].style.format({"Youth_Index": "{:.1f}%"}), use_container_width=True, hide_index=True)

# ==========================================
# TAB 2: STATE ANALYTICS
# ==========================================
with tab2:
    st.subheader("🏢 State Benchmarking")
    state_stats = filtered_df.groupby('State').agg({
        'Total_Updates': 'sum',
        'Youth_Index': 'mean' 
    }).reset_index().sort_values('Youth_Index', ascending=False)
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_state = px.bar(
            state_stats, x='State', y='Youth_Index', color='Youth_Index',
            color_continuous_scale='RdYlGn', title="State Age Profiles"
        )
        st.plotly_chart(fig_state, use_container_width=True)
    with col_b:
        st.dataframe(state_stats, use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: TREND ANALYSIS
# ==========================================
with tab3:
    st.subheader("📅 Temporal Trends")
    if trend_df is not None and not trend_df.empty:
        trend_df = trend_df.sort_values('Month_Year')
        fig_trend = px.area(
            trend_df, x='Month_Year', y=['Youth_Updates', 'Adult_Updates'],
            color_discrete_map={'Youth_Updates': '#00CC96', 'Adult_Updates': '#EF553B'}
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# ==========================================
# TAB 4: DISTRICT EXPLORER
# ==========================================
with tab4:
    st.subheader("🔎 District Intelligence")
    display_df = filtered_df.copy()
    fig_scatter = px.scatter(
        display_df, x='Total_Updates', y='Youth_Index', size='Total_Updates',
        color='State', hover_name='District', log_x=True, height=500
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)