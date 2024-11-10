# Gerekli Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram

# Veri Setini Yükle
df = pd.read_csv("datasets/Mall_Customers.csv")
print(df.head())

# Veri Kontrolü
print(df.isnull().sum())
print(df.info())

# Veri Özelliklerini Seç
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Verileri Ölçeklendir
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# K-Means ile Kümeleme
kmeans = KMeans(random_state=17)
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(X_scaled)
elbow.show()

# Optimal K Değerini Al
optimal_k = elbow.elbow_value_
print(f"Optimal K Değeri: {optimal_k}")

# Final K-Means Modelini Oluştur
kmeans_final = KMeans(n_clusters=optimal_k, random_state=17)
clusters_kmeans = kmeans_final.fit_predict(X_scaled)

# Küme Etiketlerini Veri Çerçevesine Ekle
df["KMeans Cluster"] = clusters_kmeans

# Kümeleme Sonuçlarını Görselleştir
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 1], X_scaled[:, 2], c=df["KMeans Cluster"], cmap='viridis')
plt.xlabel("Yıllık Gelir (k$)")
plt.ylabel("Harcama Puanı (1-100)")
plt.title("K-Means Kümeleme Sonuçları")
plt.show()

# Hiyerarşik Kümeleme
hc_average = linkage(X_scaled, method='average')

# Dendrogram Görselleştirmesi
plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendrogramı")
dendrogram(hc_average)
plt.show()

# Hiyerarşik Kümeleme ile Kümeleme
from sklearn.cluster import AgglomerativeClustering

n_clusters_hc = optimal_k  # İstersen burada farklı bir K değeri belirleyebilirsin
cluster_hc = AgglomerativeClustering(n_clusters=n_clusters_hc, linkage='average')
clusters_hc = cluster_hc.fit_predict(X_scaled)

# Hiyerarşik Küme Etiketlerini Veri Çerçevesine Ekle
df["Hierarchical Cluster"] = clusters_hc

# Hiyerarşik Kümeleme Sonuçlarını Görselleştir
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 1], X_scaled[:, 2], c=df["Hierarchical Cluster"], cmap='plasma')
plt.xlabel("Yıllık Gelir (k$)")
plt.ylabel("Harcama Puanı (1-100)")
plt.title("Hiyerarşik Kümeleme Sonuçları")
plt.show()
