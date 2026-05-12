
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score


df = pd.read_csv("Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)


plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans_labels)
plt.title("K-Means Clustering (Mall Customers)")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()

print("K-Means Silhouette Score:", silhouette_score(X_scaled, kmeans_labels))

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)


plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=dbscan_labels)
plt.title("DBSCAN Clustering (Mall Customers)")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()


df['Cluster'] = kmeans_labels

print("\nCluster Profiles:")
print(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())



iris = load_iris()
X = iris.data[:, :2]  # first 2 features

X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42)
labels_k = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels_k)
plt.title("K-Means (Iris)")
plt.show()

print("Iris K-Means Silhouette:", silhouette_score(X_scaled, labels_k))


dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_d = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels_d)
plt.title("DBSCAN (Iris)")
plt.show()