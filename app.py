import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="UIDAI Enrolment Analytics",
    page_icon="🆔",
    layout="wide"
)

# --- Data Loading & Cleaning ---
@st.cache_data(show_spinner="Cleaning and analyzing data...")
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
        
    try:
        # Load with low_memory=False to handle the large row count efficiently
        df = pd.read_csv(file_path, low_memory=False)
        
        # 1. Clean Dates (Your file uses DD-MM-YYYY)
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        
        # 2. Convert Numeric Columns & Handle Missing Values
        cols_to_fix = ['age_0_5', 'age_5_17', 'age_18_greater']
        for col in cols_to_fix:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 3. Create Total Column
        df['total_enrolments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
        
        # 4. Extract Time Features
        df['month'] = df['date'].dt.strftime('%B %Y')
        df['day_name'] = df['date'].dt.day_name()
        
        return df
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

# --- Main App ---
def main():
    st.title("📊 Aadhar Enrolment Insights Dashboard")
    st.markdown("---")

    # Correct filename from your upload
    FILE_NAME = "api_data_aadhar_enrolment_0_500000.csv"
    
    df = load_data(FILE_NAME)

    if df is None:
        st.error(f"⚠️ Dataset `{FILE_NAME}` not found.")
        st.info("Ensure the CSV file is in the same GitHub repository folder as this `app.py` script.")
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Global Filters")
    
    # State Filter
    states = sorted(df['state'].unique().tolist())
    selected_states = st.sidebar.multiselect("Select States", states, default=states[:5])
    
    # Date Range Filter
    valid_dates = df['date'].dropna()
    min_d, max_d = valid_dates.min().to_pydatetime(), valid_dates.max().to_pydatetime()
    date_range = st.sidebar.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

    # Apply Filters
    filtered_df = df.copy()
    if selected_states:
        filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]
    if len(date_range) == 2:
        filtered_df = filtered_df[(filtered_df['date'].dt.date >= date_range[0]) & 
                                 (filtered_df['date'].dt.date <= date_range[1])]

    # --- Metrics Bar ---
    total_e = filtered_df['total_enrolments'].sum()
    c_05 = filtered_df['age_0_5'].sum()
    c_517 = filtered_df['age_5_17'].sum()
    c_18 = filtered_df['age_18_greater'].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Registrations", f"{total_e:,.0f}")
    m2.metric("Age 0-5", f"{c_05:,.0f}", f"{c_05/total_e:.1%}" if total_e > 0 else "0%")
    m3.metric("Age 5-17", f"{c_517:,.0f}", f"{c_517/total_e:.1%}" if total_e > 0 else "0%")
    m4.metric("Age 18+", f"{c_18:,.0f}", f"{c_18/total_e:.1%}" if total_e > 0 else "0%")

    st.markdown("### Regional & Demographic Insights")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📍 Top Enrolment Districts")
        dist_data = filtered_df.groupby(['state', 'district'])['total_enrolments'].sum().reset_index().sort_values('total_enrolments', ascending=False)
        fig_dist = px.bar(dist_data.head(15), x='total_enrolments', y='district', color='state', 
                          orientation='h', labels={'total_enrolments': 'Enrolments'})
        fig_dist.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.subheader("👥 Demographic Split")
        demo_fig = px.pie(
            values=[c_05, c_517, c_18],
            names=['0-5 Years', '5-17 Years', '18+ Years'],
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(demo_fig, use_container_width=True)

    st.markdown("### 📈 Timeline Analysis")
    timeline = filtered_df.groupby('date')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
    fig_time = px.line(timeline, x='date', y=['age_0_5', 'age_5_17', 'age_18_greater'], 
                       labels={'value': 'Count', 'date': 'Date', 'variable': 'Age Group'},
                       title="Daily Registration Velocity")
    st.plotly_chart(fig_time, use_container_width=True)

    # --- Raw Data Expander ---
    with st.expander("📄 View Filtered Dataset"):
        st.write(f"Showing {len(filtered_df):,} rows")
        st.dataframe(filtered_df.head(1000), use_container_width=True)

if __name__ == "__main__":
    main()