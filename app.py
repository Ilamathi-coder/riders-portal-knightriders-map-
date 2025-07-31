import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("📊 Customer Segmentation using RFM")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # Data Cleaning
    df.dropna(subset=['InvoiceNo', 'CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # RFM Calculation
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })

    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)

    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm_scaled = rfm[['Recency', 'Frequency', 'Monetary']]
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.subheader("📊 RFM Table with Clusters")
    st.dataframe(rfm.head(10))

    # 📈 Charts
    st.subheader("📈 Cluster-wise Charts")
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.box(rfm, x='Cluster', y='Recency', title="Recency by Cluster")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(rfm, x='Cluster', y='Frequency', title="Frequency by Cluster")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.box(rfm, x='Cluster', y='Monetary', title="Monetary by Cluster")
        st.plotly_chart(fig3, use_container_width=True)

    # 📤 Export CSV
    st.subheader("📥 Export Segmented Data")
    csv = rfm.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📤 Download Segmented CSV",
        data=csv,
        file_name='segmented_customers.csv',
        mime='text/csv',
    )
