# Customer Segmentation Project
# Thiranex Assignment

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------
# 1. Create customer dataset
# ---------------------------------------

data = {
    "CustomerID": range(1, 21),

    "Age": [
        19, 21, 20, 23, 31,
        22, 35, 40, 28, 30,
        45, 25, 32, 38, 27,
        50, 29, 41, 36, 24
    ],

    "Annual Income (k$)": [
        15, 16, 17, 18, 20,
        25, 35, 40, 45, 50,
        60, 65, 70, 75, 80,
        85, 90, 95, 100, 105
    ],

    "Spending Score (1-100)": [
        39, 81, 6, 77, 40,
        76, 6, 20, 55, 65,
        30, 73, 45, 10, 85,
        20, 75, 35, 90, 70
    ]
}

df = pd.DataFrame(data)


# ---------------------------------------
# 2. Display dataset
# ---------------------------------------

print("FIRST 5 CUSTOMERS")
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())

print("\nDATASET DESCRIPTION")
print(df.describe())

print("\nDATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum())


# ---------------------------------------
# 3. Select features
# ---------------------------------------

X = df[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
]


# ---------------------------------------
# 4. Standardize data
# ---------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ---------------------------------------
# 5. Elbow Method
# ---------------------------------------

inertia = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.grid(True)

plt.show()


# ---------------------------------------
# 6. Apply K-Means Clustering
# ---------------------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# ---------------------------------------
# 7. Display customer segments
# ---------------------------------------

print("\nCUSTOMER SEGMENTS")

print(df)


# ---------------------------------------
# 8. Analyze each cluster
# ---------------------------------------

print("\nAVERAGE CHARACTERISTICS OF EACH CLUSTER")

cluster_analysis = df.groupby("Cluster")[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].mean()

print(cluster_analysis)


# ---------------------------------------
# 9. Customer Segmentation Graph
# ---------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    s=100
)

plt.title("Customer Segmentation")

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.legend(title="Customer Cluster")

plt.show()


# ---------------------------------------
# 10. Number of customers in each cluster
# ---------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Cluster"
)

plt.title("Number of Customers in Each Cluster")

plt.xlabel("Customer Cluster")

plt.ylabel("Number of Customers")

plt.show()


# ---------------------------------------
# 11. Display final result
# ---------------------------------------

print("\nFINAL CUSTOMER SEGMENTATION")

for cluster in sorted(df["Cluster"].unique()):

    count = len(
        df[df["Cluster"] == cluster]
    )

    print(
        "Cluster",
        cluster,
        ":",
        count,
        "customers"
    )


print("\nCustomer Segmentation Completed Successfully!")

