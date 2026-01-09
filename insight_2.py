import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Try importing Groq, handle if missing
try:
    from groq import Groq
    groq_available = True
except ImportError:
    groq_available = False

# --- 1. PAGE CONFIGURATION & THEME ---
st.set_page_config(
    page_title="Aadhar Enrolment Intelligence",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. ADVANCED STYLING (The "Demographic Dividend" Theme Applied) ---
st.markdown("""
<style>
    /* Professional Dark Gradient Background */
    .stApp {
        background: radial-gradient(circle at top center, #1e293b 0%, #0f172a 100%);
        background-attachment: fixed;
        color: #f1f5f9;
    }

    /* GLASSMORPHISM CARDS */
    .metric-card {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00CC96;
    }
    
    /* Typography */
    h1, h2, h3, h4 { 
        font-family: 'Helvetica Neue', sans-serif; 
        color: #f1f5f9;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .h1-title {
        text-align: center;
        font-weight: 900;
        font-size: 2.8em;
        background: -webkit-linear-gradient(left, #00CC96, #3366FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
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
    
    /* Stat Highlights */
    .big-stat {
        font-size: 3em;
        font-weight: 800;
        color: #00CC96;
        line-height: 1;
    }
    .stat-label {
        color: #a0a0a0;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Detailed Explanation Box */
    .explanation-box {
        background: rgba(0, 0, 0, 0.3);
        border-left: 5px solid #3366FF;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        font-size: 1em;
        color: #e0e0e0;
        line-height: 1.6;
    }

    /* --- FLOATING CHATBOT BUTTON (ULTRA MODERN FAB) --- */
    div[data-testid="stPopover"] {
        position: fixed;
        bottom: 40px;
        right: 40px;
        z-index: 9999;
        width: auto;
    }

    div[data-testid="stPopover"]::before {
        content: "✨ AI Operations Analyst";
        position: absolute;
        top: 50%;
        right: 100%;
        transform: translateY(-50%);
        margin-right: 20px;
        width: max-content;
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid #334155;
        color: #00bfff;
        padding: 8px 14px;
        border-radius: 10px;
        font-size: 13px;
        font-weight: bold;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        animation: tooltipFloat 5s ease-in-out infinite;
        pointer-events: none;
        z-index: 10001;
    }

    @keyframes tooltipFloat {
        0%, 100% { opacity: 0; transform: translateY(-50%) translateX(10px); }
        10%, 90% { opacity: 1; transform: translateY(-50%) translateX(0); }
    }
    
    div[data-testid="stPopover"] > button {
        width: 90px !important;
        height: 90px !important;
        border-radius: 50% !important;
        background: linear-gradient(300deg, #00bfff, #00CC96, #8A2BE2) !important;
        background-size: 200% 200% !important;
        animation: gradientBG 4s ease infinite, float 3s ease-in-out infinite !important;
        box-shadow: 0 10px 25px rgba(0, 204, 150, 0.5), inset 0 0 10px rgba(255,255,255,0.2) !important;
        border: none !important;
        color: white !important;
        font-size: 40px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }
    
    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.15) rotate(10deg) !important;
        box-shadow: 0 15px 35px rgba(0, 191, 255, 0.6), 0 0 15px rgba(255,255,255,0.4) !important;
        cursor: pointer;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }

    div[data-testid="stPopoverBody"] {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background-color: rgba(15, 23, 42, 0.95);
        box-shadow: 0 20px 50px rgba(0,0,0,0.6);
        padding: 0 !important;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. GROQ AI CLIENT SETUP ---
def get_groq_client():
    if not groq_available:
        return None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except KeyError:
        return None

client = get_groq_client()

# --- 4. DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_process_data():
    files = [
        "api_data_aadhar_enrolment_0_500000.csv",
        "api_data_aadhar_enrolment_500000_1000000.csv",
        "api_data_aadhar_enrolment_1000000_1006029.csv"
    ]
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except FileNotFoundError:
            return None

    if not dfs:
        return None
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Robust Date Parsing
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    
    # Key Metrics
    df['total_enrolment'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    
    # Standardization
    state_mapping = {
        'Westbengal': 'West Bengal', 'West  Bengal': 'West Bengal', 
        'West Bangal': 'West Bengal', 'Andhra Pradesh': 'Andhra Pradesh',
        'andhra pradesh': 'Andhra Pradesh', 'Odisha': 'Odisha',
        'ODISHA': 'Odisha', 'Orissa': 'Odisha', 'Pondicherry': 'Puducherry'
    }
    df['state'] = df['state'].str.strip().str.title().replace(state_mapping)
    df = df[df['state'] != '100000']
    
    # Feature Engineering (The Core Insight)
    df['Era'] = df['date'].apply(lambda x: 'Real-Time Era (Sept+)' if (x.year == 2025 and x.month >= 9) else 'Batch Era (Pre-Aug)')
    df['DayOfWeek'] = df['date'].dt.day_name()
    
    return df

df = load_and_process_data()

# --- 5. AI CONTEXT PREPARATION ---
def prepare_data_context(df_filtered, era_selection, state_selection, growth_pct):
    """Summarizes operational efficiency data for the AI"""
    if df_filtered.empty:
        return "No data available."
    
    total_vol = df_filtered['total_enrolment'].sum()
    row_count = len(df_filtered)
    
    # Top 3 High Activity Districts
    top_districts = df_filtered.groupby('district')['total_enrolment'].sum().nlargest(3).to_dict()
    
    # Meghalaya 18+ Stats (Specific Insight)
    meghalaya_stats = "N/A"
    if 'Meghalaya' in df_filtered['state'].unique():
        meg_df = df_filtered[df_filtered['state'] == 'Meghalaya']
        meg_18 = (meg_df['age_18_greater'].sum() / meg_df['total_enrolment'].sum()) * 100
        meghalaya_stats = f"{meg_18:.1f}% Adult Enrolment (High)"

    context = f"""
    DASHBOARD CONTEXT:
    - User Filter Selection: Eras={era_selection}, States={state_selection}
    - Total Enrolments: {total_vol:,}
    - Operational Activity (Rows Processed): {row_count:,}
    - CALCULATED EFFICIENCY GROWTH: {growth_pct:.1f}% (July Batch vs Sept Real-time)
    - Key Insight: Shift from Batch Processing to Real-Time API Logging.
    - Top Districts: {top_districts}
    - Meghalaya Anomaly: {meghalaya_stats}
    """
    return context

def get_ai_response(messages):
    if not client:
        return "⚠️ AI Features Unavailable: GROQ_API_KEY missing or groq library not installed."
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- 6. MAIN DASHBOARD ---
if df is not None:
    
    # --- HEADER ---
    st.markdown("<h1 class='h1-title'>National Aadhar Analytics</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #cbd5e1; margin-bottom: 30px;'>
        Advanced Forensic Analysis: Uncovering the shift from Batch Processing to Real-Time Operational Intelligence.
    </div>
    """, unsafe_allow_html=True)

    # --- CONTROLS ---
    st.markdown("### 🎛️ Strategic Filters")
    col_filter_1, col_filter_2 = st.columns(2)
    with col_filter_1:
        selected_era = st.multiselect("Reporting Era:", df['Era'].unique(), default=df['Era'].unique())
    with col_filter_2:
        states = sorted(df['state'].unique())
        selected_states = st.multiselect("Focus States:", states, default=states[:5])

    df_filtered = df[df['Era'].isin(selected_era)]
    if selected_states:
        df_filtered = df_filtered[df_filtered['state'].isin(selected_states)]

    st.markdown("---")

    # --- KPI SECTION ---
    july_vol = df[df['date'].dt.month == 7]['total_enrolment'].sum() / 30
    sept_vol = df[df['date'].dt.month == 9].groupby('date')['total_enrolment'].sum().mean()
    growth = ((sept_vol - july_vol) / july_vol) * 100
    top_state_growth = df_filtered.groupby('state')['total_enrolment'].sum().idxmax()

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="stat-label">Efficiency Growth</div>
            <div class="big-stat">+{growth:.1f}%</div>
            <div style="color: #4CAF50;">▲ True Daily Run Rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="stat-label">Processing Power</div>
            <div class="big-stat">{int(sept_vol):,}</div>
            <div style="color: #cbd5e1;">Daily Transactions (Sept)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="stat-label">Volume Leader</div>
            <div class="big-stat" style="font-size: 2em;">{top_state_growth}</div>
            <div style="color: #cbd5e1;">Top Contributor</div>
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION 1: CORE INSIGHT ---
    st.markdown("### 1. The Operational Shift")
    st.markdown("""
    <div class='explanation-box'>
        <b>The Insight:</b> The visual divergence below proves the system didn't crash; it evolved.
        <ul>
            <li><b>Red Area (Volume):</b> Drops because we stopped uploading massive monthly batches.</li>
            <li><b>Green Line (Activity):</b> Spikes because we started logging every single daily transaction.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    daily_agg = df.groupby('date').agg({'total_enrolment': 'sum', 'state': 'count'}).rename(columns={'state': 'row_count'})
    
    fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
    fig_combo.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['total_enrolment'], name="Enrolment Volume",
                                   line=dict(color='#ff4b4b', width=2), fill='tozeroy', fillcolor='rgba(255, 75, 75, 0.1)'), secondary_y=False)
    fig_combo.add_trace(go.Scatter(x=daily_agg.index, y=daily_agg['row_count'], name="System Activity (Rows)",
                                   line=dict(color='#00CC96', width=3, dash='solid')), secondary_y=True)
    
    fig_combo.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=1.1),
        height=450,
        margin=dict(l=0,r=0,t=40,b=0)
    )
    st.plotly_chart(fig_combo, use_container_width=True)

    # --- SECTION 2: TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Hierarchy", "🏭 Growth Engines", "🗓️ Heatmap", "🚨 Anomalies"])
    
    with tab1:
        st.caption("Drill Down: State > District > Age Group")
        df_melted = df_filtered.melt(id_vars=['state', 'district'], value_vars=['age_0_5', 'age_5_17', 'age_18_greater'], var_name='Age_Group', value_name='Count')
        sunburst_data = df_melted.groupby(['state', 'district', 'Age_Group'])['Count'].sum().reset_index()
        sunburst_data = sunburst_data[sunburst_data['Count'] > 0]
        fig_sun = px.sunburst(sunburst_data, path=['state', 'district', 'Age_Group'], values='Count', color='Count', color_continuous_scale='Viridis')
        fig_sun.update_layout(height=600, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sun, use_container_width=True)

    with tab2:
        st.caption("Top 10 High-Velocity Districts (Real-Time Era Only)")
        rt_df = df[df['Era'] == 'Real-Time Era (Sept+)']
        if not rt_df.empty:
            district_growth = rt_df.groupby(['state', 'district'])['total_enrolment'].sum().reset_index().sort_values('total_enrolment', ascending=False).head(10)
            fig_bar = px.bar(district_growth, x='total_enrolment', y='district', color='state', orientation='h', text='total_enrolment')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Enable 'Real-Time Era' filter to see this.")

    with tab3:
        st.caption("Operational Rhythm: Day vs State")
        heatmap_data = df_filtered.groupby(['state', 'DayOfWeek'])['total_enrolment'].sum().reset_index()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig_heat = px.density_heatmap(heatmap_data, x='DayOfWeek', y='state', z='total_enrolment', category_orders={'DayOfWeek': days_order}, color_continuous_scale='Hot')
        fig_heat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=600)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab4:
        st.caption("Strategic Outlier: Meghalaya (High Adult Enrolment)")
        state_stats = df.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater', 'total_enrolment']].sum()
        state_stats['pct_18_plus'] = (state_stats['age_18_greater'] / state_stats['total_enrolment']) * 100
        state_stats = state_stats.sort_values('pct_18_plus', ascending=False).head(10).reset_index()
        colors = ['#ff4b4b' if x == 'Meghalaya' else '#2c5364' for x in state_stats['state']]
        fig_ano = px.bar(state_stats, x='state', y='pct_18_plus', text='pct_18_plus')
        fig_ano.update_traces(marker_color=colors, texttemplate='%{text:.1f}%')
        fig_ano.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_ano, use_container_width=True)

    # --- 7. FLOATING AI CHATBOT (FAB) ---
    if groq_available:
        with st.popover("✨", use_container_width=False):
            # Sticky Header inside Popover
            st.markdown(
                """
                <div style="
                    position: sticky; top: 0; background-color: #0f172a; z-index: 1000; padding: 15px 10px; border-bottom: 1px solid #334155; margin: -1rem; margin-bottom: 10px;
                ">
                    <div style="display: flex; align-items: center; gap: 10px; padding-left: 10px;">
                        <span style="font-size: 24px;">🤖</span>
                        <div>
                            <h3 style="margin: 0; font-size: 16px; color: #f1f5f9;">Ops Intelligence</h3>
                            <p style="margin: 0; font-size: 11px; color: #94a3b8;">Real-Time Analyst</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Prepare Dynamic Context based on User's current filters
            data_ctx = prepare_data_context(df_filtered, selected_era, selected_states, growth)

            system_prompt = {
                "role": "system",
                "content": f"""You are an Expert Operational Analyst for the National Aadhar Enrolment Project.
                
                CORE INSIGHT TO DEFEND:
                The data shows a massive drop in VOLUME in September. This is NOT a failure.
                It is because we switched from 'Monthly Batch Uploads' (July) to 'Real-Time Daily API' (Sept).
                Operational Efficiency (Row Count) actually INCREASED by {growth:.1f}%.
                
                CURRENT DASHBOARD DATA:
                {data_ctx}
                
                Instructions:
                - Provide SPECIFIC policy recommendations for the government based on the data.
                - Defend the 'Efficiency Growth' narrative.
                - Use the provided context numbers to answer questions.
                - If asked about anomalies, mention Meghalaya's high adult enrolment.
                - Keep answers short, professional, and data-backed.
                """
            }

            # Auto-Greet and Policy Measures
            if not st.session_state.chat_history:
                initial_user_prompt = "Give me a list of recommended government policy measures based on this data."
                st.session_state.chat_history.append({"role": "user", "content": initial_user_prompt})
                
                # Pre-generate response to ensure immediate display
                full_msgs = [system_prompt] + st.session_state.chat_history
                with st.spinner("Generating Policy Recommendations..."):
                    initial_response = get_ai_response(full_msgs)
                
                st.session_state.chat_history.append({"role": "assistant", "content": initial_response})


            # Chat Interface
            chat_container = st.container(height=350)
            with chat_container:
                for message in st.session_state.chat_history:
                    if message["role"] != "system":
                         # Hide the auto-triggered user prompt from view to make it look like the bot started it
                        if message["content"] == "Give me a list of recommended government policy measures based on this data.":
                             continue
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

            if prompt := st.chat_input("Ask about the data...", key="fab_chat"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                full_msgs = [system_prompt] + st.session_state.chat_history
                
                with st.spinner("Processing..."):
                    response = get_ai_response(full_msgs)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

else:
    st.error("Data files not found. Please upload the CSV files.")