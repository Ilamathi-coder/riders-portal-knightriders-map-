import pandas as pd
import datetime as dt

# Load the dataset
df = pd.read_excel("OnlineRetail.xlsx")

# Drop rows with missing CustomerID
df.dropna(subset=['CustomerID'], inplace=True)

# Keep only positive quantities and prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Create TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Convert InvoiceDate to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Show cleaned data
print("Cleaned Data:\n", df.head())
# STEP 5: Create RFM Table

# Step 5.1: Create a reference date (1 day after last invoice)
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Step 5.2: Group by CustomerID to calculate RFM
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,   # Recency
    'InvoiceNo': 'nunique',                                     # Frequency
    'TotalPrice': 'sum'                                         # Monetary
})

# Step 5.3: Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Step 5.4: Display top 5 rows
print("\nRFM Table:")
print(rfm.head())
# --- STEP 6: Feature Scaling ---
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

print("\nScaled RFM values ready for clustering.")
# --- STEP 7: K-Means Clustering ---
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print("\nRFM with Cluster Labels:\n", rfm.head())
# --- STEP 8: Analyze the clusters ---
cluster_summary = rfm.groupby('Cluster').mean()
print("\nAverage RFM values per cluster:\n", cluster_summary)
# --- STEP 9: Visualize the Clusters ---
import matplotlib.pyplot as plt
import seaborn as sns

rfm_plot = rfm.reset_index()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='Recency', data=rfm_plot)
plt.title("Recency by Cluster")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='Frequency', data=rfm_plot)
plt.title("Frequency by Cluster")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='Monetary', data=rfm_plot)
plt.title("Monetary by Cluster")
plt.show()
