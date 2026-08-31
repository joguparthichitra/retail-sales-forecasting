import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")

st.title("Retail Sales Forecasting Dashboard")
st.markdown("XGBoost-based sales forecasting for Rossmann retail stores")

@st.cache_resource
def load_model():
    return joblib.load('xgboost_sales_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('test_processed.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

with open('feature_cols.json', 'r') as f:
    feature_cols = json.load(f)

model = load_model()
test_df = load_data()

st.sidebar.header("Filters")
store_list = sorted(test_df['Store'].unique())
selected_store = st.sidebar.selectbox("Select Store", store_list)

store_data = test_df[test_df['Store'] == selected_store].sort_values('Date')

X_store = store_data[feature_cols]
predictions = model.predict(X_store)
store_data = store_data.copy()
store_data['Predicted_Sales'] = predictions

col1, col2, col3 = st.columns(3)
mae = np.mean(np.abs(store_data['Sales'] - store_data['Predicted_Sales']))
mape = np.mean(np.abs((store_data['Sales'] - store_data['Predicted_Sales']) / store_data['Sales'])) * 100
avg_sales = store_data['Sales'].mean()

col1.metric("Avg Daily Sales", f"{avg_sales:,.0f}")
col2.metric("MAE", f"{mae:,.0f}")
col3.metric("MAPE", f"{mape:.2f}%")

st.subheader("Actual vs Predicted Sales")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(store_data['Date'], store_data['Sales'], label='Actual', marker='o')
ax.plot(store_data['Date'], store_data['Predicted_Sales'], label='Predicted', marker='x')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Feature Importance")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.barh(importance_df['feature'][::-1], importance_df['importance'][::-1])
ax2.set_xlabel('Importance')
st.pyplot(fig2)

st.subheader("Raw Data")
st.dataframe(store_data[['Date', 'Sales', 'Predicted_Sales', 'Promo', 'IsHoliday']])