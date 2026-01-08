import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Aadhar Enrolment Insights",
    page_icon="🆔",
    layout="wide"
)

# --- Helper Functions ---

@st.cache_data
def load_and_clean_data(file_path):
    """Loads data and performs cleaning/feature engineering."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

    # 1. Clean Dates
    # The snippet shows DD-MM-YYYY format
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    
    # 2. Handle missing values
    # Fill numeric columns with 0 if null, drop rows with missing critical labels if necessary
    numeric_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 3. Feature Engineering
    df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    df['month_year'] = df['date'].dt.to_period('M').astype(str)
    df['day_of_week'] = df['date'].dt.day_name()
    
    return df

# --- Main Application ---

def main():
    st.title("📊 Aadhar Enrolment Analytics Dashboard")
    st.markdown("""
    This dashboard provides deep insights into Aadhar enrolment patterns across various states and age groups. 
    Use the sidebar to filter data and explore specific regions or timeframes.
    """)

    # Load Data
    data_file = "api_data_aadhar_enrolment_0_500000.csv"
    df = load_and_clean_data(data_file)

    if df is None:
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Controls")
    
    # Date Range Filter
    min_date = df['date'].min().to_pydatetime()
    max_date = df['date'].max().to_pydatetime()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # State Filter
    all_states = sorted(df['state'].unique().tolist())
    selected_states = st.sidebar.multiselect("Select States", all_states, default=all_states[:5])

    # Filtered Dataframe
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        if selected_states:
            mask = mask & (df['state'].isin(selected_states))
        filtered_df = df.loc[mask]
    else:
        filtered_df = df

    # --- Top Level Metrics ---
    total_e = filtered_df['total_enrolments'].sum()
    child_e = filtered_df['age_0_5'].sum()
    youth_e = filtered_df['age_5_17'].sum()
    adult_e = filtered_df['age_18_greater'].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Enrolments", f"{total_e:,}")
    m2.metric("Children (0-5)", f"{child_e:,}", f"{child_e/total_e:.1%}")
    m3.metric("Youth (5-17)", f"{youth_e:,}", f"{youth_e/total_e:.1%}")
    m4.metric("Adults (18+)", f"{adult_e:,}", f"{adult_e/total_e:.1%}")

    st.divider()

    # --- Layout for Insights ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📍 Regional Distribution (Top States)")
        state_data = filtered_df.groupby('state')['total_enrolments'].sum().sort_values(ascending=False).reset_index()
        fig_state = px.bar(
            state_data.head(15), 
            x='total_enrolments', 
            y='state', 
            orientation='h',
            color='total_enrolments',
            color_continuous_scale='Viridis',
            labels={'total_enrolments': 'Enrolments', 'state': 'State'}
        )
        fig_state.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_state, use_container_width=True)

    with col_right:
        st.subheader("👥 Age Demographics Breakdown")
        demo_counts = {
            'Age 0-5': child_e,
            'Age 5-17': youth_e,
            'Age 18+': adult_e
        }
        fig_pie = px.pie(
            names=list(demo_counts.keys()), 
            values=list(demo_counts.values()),
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # --- Time Series Analysis ---
    st.subheader("📈 Enrolment Trends Over Time")
    trend_data = filtered_df.groupby('date')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['age_0_5'], name='0-5 Years', line=dict(width=2)))
    fig_line.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['age_5_17'], name='5-17 Years', line=dict(width=2)))
    fig_line.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['age_18_greater'], name='18+ Years', line=dict(width=2)))
    
    fig_line.update_layout(
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # --- District Deep Dive ---
    st.divider()
    st.subheader("🔍 District-Level Hotspots")
    
    if selected_states:
        # Show districts for the first selected state for clarity
        target_state = selected_states[0]
        st.info(f"Showing Top Districts in **{target_state}**")
        district_data = filtered_df[filtered_df['state'] == target_state].groupby('district')['total_enrolments'].sum().sort_values(ascending=False).head(10).reset_index()
        
        fig_dist = px.treemap(
            district_data, 
            path=['district'], 
            values='total_enrolments',
            color='total_enrolments',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.warning("Please select at least one state in the sidebar to view district insights.")

    # --- Data Table Section ---
    with st.expander("📄 View Raw Processed Data"):
        st.dataframe(filtered_df.sort_values(by='date', ascending=False), use_container_width=True)
        
    # --- Actionable Insights ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Key Observations")
    
    max_state = state_data.iloc[0]['state'] if not state_data.empty else "N/A"
    dominant_age = max(demo_counts, key=demo_counts.get)
    
    st.sidebar.info(f"""
    - **Top State:** {max_state} leads in total enrolments.
    - **Primary Target:** The {dominant_age} group represents the largest segment of current enrolments.
    - **Trend Check:** View the line chart to identify if enrolment is peaking or declining.
    """)

if __name__ == "__main__":
    main()