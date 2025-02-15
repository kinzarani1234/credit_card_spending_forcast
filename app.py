import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np
import gdown

# -------------------------------
# 1. Data Loading & Caching
# -------------------------------
@st.cache_data
def load_data():
    # Google Drive file ID and direct download URL construction
    file_id = "1B91ZsneAb5lK7PUM_Eoaa3vCZEn010K5"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "credit_card_transactions.csv"
    
    # Download the file using gdown (quiet mode set to False for progress)
    gdown.download(url, output, quiet=True)
    
    # Read the CSV file
    df = pd.read_csv(output)
    
    # Rename date column if necessary and convert to datetime
    if 'trans_date_trans_time' in df.columns:
        df.rename(columns={'trans_date_trans_time': 'transaction_date'}, inplace=True)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    return df

df = load_data()

# -------------------------------
# 2. Sidebar: User Input Panel
# -------------------------------
st.sidebar.title("User Input Panel")

# Select Customer ID
customer_ids = df['cc_num'].unique()
selected_customer = st.sidebar.selectbox("Select Customer ID", customer_ids)

# Select Spending Category
categories = df['category'].unique()
selected_category = st.sidebar.selectbox("Select Spending Category", categories)

# Select Time Range
min_date = df['transaction_date'].min().date()
max_date = df['transaction_date'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range", 
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Forecast Horizon: 3 or 6 months (only applicable for daily aggregation)
forecast_horizon = st.sidebar.selectbox("Forecast Horizon (months)", [3, 6])
forecast_days = forecast_horizon * 30  # Approximate number of days

# -------------------------------
# 3. Data Filtering & Aggregation
# -------------------------------
mask = (
    (df['cc_num'] == selected_customer) &
    (df['category'] == selected_category) &
    (df['transaction_date'] >= pd.to_datetime(date_range[0])) &
    (df['transaction_date'] <= pd.to_datetime(date_range[1]))
)
filtered_df = df.loc[mask].copy()

st.write("### Filtered Transaction Data", filtered_df.head())

if not filtered_df.empty:
    # Check if the selected date range is a single day (or less than one day)
    date_diff = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days
    if date_diff < 1:
        # ----- Hourly Aggregation -----
        # Use the existing 'hour' column if available; otherwise extract from transaction_date.
        if 'hour' not in filtered_df.columns:
            filtered_df['hour'] = filtered_df['transaction_date'].dt.hour

        # Aggregate spending by hour
        hourly_data = filtered_df.groupby('hour')['amt'].sum().reset_index()
        # Create a formatted hour label (e.g., "1 AM", "2 PM")
        hourly_data['hour_label'] = hourly_data['hour'].apply(lambda h: f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}")
        # For Prophet, create a datetime column using the selected date and the hour value.
        selected_date = pd.to_datetime(date_range[0])
        hourly_data['ds'] = hourly_data['hour'].apply(lambda h: selected_date.replace(hour=int(h), minute=0, second=0))
        hourly_data.rename(columns={'amt': 'y'}, inplace=True)

        st.write("### Historical Spending Trend (Hourly)")
        st.line_chart(hourly_data.set_index('hour_label')['y'])
        
        # Forecast hourly spending for the next 24 hours
        m = Prophet(daily_seasonality=False, yearly_seasonality=False)
        m.fit(hourly_data[['ds', 'y']])
        future = m.make_future_dataframe(periods=24, freq='H')
        forecast = m.predict(future)
        
        st.write("### Forecasted Spending Trend (Hourly)")
        fig1 = m.plot(forecast)
        plt.title("Hourly Forecast for Next 24 Hours")
        st.pyplot(fig1)
        
        st.write("### Forecast Components (Hourly)")
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)
    else:
        # ----- Daily Aggregation -----
        daily_data = filtered_df.groupby('transaction_date')['amt'].sum().reset_index()
        daily_data.rename(columns={'transaction_date': 'ds', 'amt': 'y'}, inplace=True)
        
        st.write("### Historical Spending Trend (Daily)")
        st.line_chart(daily_data.set_index('ds'))
        
        m = Prophet()
        m.fit(daily_data)
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        
        st.write("### Forecasted Spending Trend (Daily)")
        fig1 = m.plot(forecast)
        plt.title("Forecast of Spending for Next {} Months".format(forecast_horizon))
        st.pyplot(fig1)
        
        st.write("### Forecast Components (Daily)")
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)
else:
    st.write("No data available for the selected filters.")

# -------------------------------
# 4. Expected Outcomes (Display)
# -------------------------------
st.write("### Expected Outcomes")
st.markdown("""
- **Improved understanding** of customer spending patterns.
- **Accurate predictions** of future credit card spending.
- A **user-friendly dashboard** for financial institutions & individuals.
- Potential integration with banks for **real-time forecasting** & **budget recommendations**.
""")
