import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime as dt

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🧠 Customer Segmentation Using RFM + K-Means")

uploaded_file = st.file_uploader("📁 Upload OnlineRetail Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Data Cleaning
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    st.subheader("✅ Cleaned Data Preview")
    st.dataframe(df.head())

    # RFM Calculation
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.subheader("📊 RFM Table with Clusters")
    st.dataframe(rfm.reset_index())

    # Cluster Summary
    st.subheader("📈 Average RFM by Cluster")
    st.dataframe(rfm.groupby('Cluster').mean())

    # Charts
    st.subheader("📉 Visualizations")

    fig1 = plt.figure(figsize=(10, 4))
    sns.boxplot(x='Cluster', y='Recency', data=rfm.reset_index())
    st.pyplot(fig1)

    fig2 = plt.figure(figsize=(10, 4))
    sns.boxplot(x='Cluster', y='Frequency', data=rfm.reset_index())
    st.pyplot(fig2)

    fig3 = plt.figure(figsize=(10, 4))
    sns.boxplot(x='Cluster', y='Monetary', data=rfm.reset_index())
    st.pyplot(fig3)

    # Export Option
    st.subheader("📤 Export Segmented Data")
    csv = rfm.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "segmented_customers.csv", "text/csv")

else:
    st.info("📎 Please upload the dataset to begin.")
