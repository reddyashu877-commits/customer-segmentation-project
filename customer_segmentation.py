# Customer Segmentation Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")

# 2. Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# 3. Select features for clustering
# Age, Annual Income and Spending Score
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]


# 4. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Elbow Method
inertia = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()


# 6. Create 5 customer clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# 7. Display customers with clusters
print("\nCustomer Segments:")
print(df.head(20))


# 8. Cluster analysis
print("\nAverage Characteristics of Each Cluster:")
print(
    df.groupby("Cluster")[
        ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    ].mean()
)


# 9. Visualize customer segments
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Customer Cluster")
plt.show()


# 10. Count customers in each cluster
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Cluster"
)

plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.show()


# 11. Save the result
df.to_csv("customer_segments.csv", index=False)

print("\nCustomer segmentation completed successfully!")
print("Result saved as customer_segments.csv")
