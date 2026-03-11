import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.set_page_config(page_title="GridPulse: Demand Forecaster", layout="wide")

@st.cache_data 
def load_and_train():
    df = pd.read_excel("demo.xlsx")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['Demand_Lag_24h'] = df['Demand (MW)'].shift(24)
    df = df.dropna()

    features = ['Hour', 'DayOfWeek', 'Month', 'Solar(MW)','Wind(MW)', 'Demand_Lag_24h']
    
    # Split for the graph data (Jan-May Train, June Test)
    train_data = df[df['Month'] <= 5]
    test_data = df[df['Month'] == 6]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_data[features], train_data['Demand (MW)'])
    
    # Generate June predictions for the graph
    june_preds = model.predict(test_data[features])
    
    return model, df['Demand (MW)'].iloc[-1], test_data['Demand (MW)'], june_preds

# Unpack the new return values
model, last_known_demand, y_test, predictions = load_and_train()

# --- SIDEBAR ---
st.sidebar.header("🕹️ Control Panel")
input_date = st.sidebar.date_input("Select Date", value=pd.to_datetime("2025-06-25"))
input_hour = st.sidebar.slider("Select Hour", 0, 23, 12)
solar_input = st.sidebar.slider("Solar Generation (MW)", 0, 50000, 15000)
wind_input = st.sidebar.slider("Wind Generation (MW)", 0, 50000, 10000)

# --- MAIN PAGE ---
st.title(" GridPulse: Real-Time Demand Forecasting")
st.markdown("Predicting Indian grid demand using behavioral patterns and renewable proxies.")

# 1. Single Prediction Calculation
input_data = pd.DataFrame({
    'Hour': [input_hour],
    'DayOfWeek': [input_date.weekday()],
    'Month': [input_date.month],
    'Solar(MW)': [solar_input],
    'Wind(MW)': [wind_input],
    'Demand_Lag_24h': [last_known_demand]
})
prediction = model.predict(input_data)[0]

# 2. Key Metrics
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Predicted Demand", value=f"{prediction:,.2f} MW")
with col2:
    status = "Normal Load" if prediction < 210000 else "High Demand Warning"
    st.metric(label="System Status", value=status, 
              delta="Alert" if prediction > 210000 else None, delta_color="inverse")

st.info(f"💡 Analysis: At {input_hour}:00, the grid expects a load of {prediction:,.0f} MW.")

# --- 3. INTERACTIVE GRAPH SECTION ---
st.divider()
st.subheader("📈 Historical Accuracy (June 2025 Performance)")
st.write("Zoom and hover over the chart to compare actual demand vs. model predictions.")

# Prepare the data for Plotly (First 168 hours of June)
plot_df = pd.DataFrame({
    'Hour Index': range(168),
    'Actual Demand': y_test.values[:168],
    'Model Prediction': predictions[:168]
}).melt(id_vars='Hour Index', var_name='Type', value_name='Demand (MW)')



fig = px.line(plot_df, x='Hour Index', y='Demand (MW)', color='Type',
              color_discrete_map={'Actual Demand': '#1f77b4', 'Model Prediction': '#ffa500'},
              template="plotly_white")

fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)