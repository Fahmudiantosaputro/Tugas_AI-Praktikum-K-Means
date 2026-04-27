📊 Customer Segmentation menggunakan K-Means (Python)
📌 Deskripsi Project

Project ini bertujuan untuk melakukan segmentasi pelanggan (customer segmentation) menggunakan algoritma K-Means Clustering.

Dengan metode ini, kita bisa mengelompokkan pelanggan berdasarkan:

Pendapatan Tahunan (Annual Income)
Skor Pengeluaran (Spending Score)

Hasilnya bisa digunakan untuk:

strategi marketing
memahami perilaku customer
meningkatkan penjualan
🧰 Tools & Library

Project ini menggunakan beberapa library Python:

pandas → mengelola data
numpy → perhitungan numerik
matplotlib → visualisasi grafik
seaborn → visualisasi lebih menarik
sklearn → algoritma K-Means
📂 Dataset

Dataset yang digunakan adalah:
Mall Customers Dataset

Berisi informasi:

Customer ID
Gender
Age
Annual Income
Spending Score

Namun pada project ini hanya digunakan:

Annual Income
Spending Score
🚀 Langkah-langkah Pengerjaan
1️⃣ Import Library

Mengimpor semua library yang dibutuhkan.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
2️⃣ Import Dataset

Membaca file dataset.

dataset = pd.read_csv("Mall_Customers.csv")
dataset.head()
3️⃣ Mengambil Fitur

Hanya mengambil 2 variabel agar bisa divisualisasikan dalam 2D.

X = dataset.iloc[:, 3:5]
4️⃣ Eksplorasi Data

Melihat kondisi data:

X.shape
X.isnull().sum()
X.describe()

Penjelasan:

Tidak ada data kosong
Data siap digunakan
5️⃣ Menentukan Jumlah Cluster (Metode Elbow)

Digunakan untuk mencari jumlah cluster terbaik.

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, random_state=14)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 15), wcss)
plt.title("Metode Elbow")
plt.xlabel("Jumlah Cluster")
plt.ylabel("WCSS")
plt.show()

📌 Hasil:

Titik “siku” ada di k = 5
Maka digunakan 5 cluster
6️⃣ Proses Clustering

Melakukan clustering dengan K-Means.

kmeans = KMeans(n_clusters=5, random_state=14)
kmeans.fit(X)
7️⃣ Menambahkan Label Cluster ke Data
hasil_kmeans = X.copy()
hasil_kmeans["cluster"] = kmeans.labels_
8️⃣ Visualisasi Frekuensi Cluster

Menampilkan jumlah data di setiap cluster.

cluster_counts = hasil_kmeans["cluster"].value_counts().sort_index()

sns.barplot(
    x=cluster_counts.index,
    y=cluster_counts.values,
    hue=cluster_counts.index,
    legend=False
)

plt.title("Frekuensi Data per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Jumlah Data")
plt.show()
9️⃣ Visualisasi Scatter Plot

Menampilkan hasil clustering dalam bentuk grafik.

centroid = kmeans.cluster_centers_

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=kmeans.labels_)

plt.scatter(
    centroid[:,0],
    centroid[:,1],
    s=200,
    c="black",
    label="Centroid"
)

plt.title("Cluster Customer")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
📊 Hasil & Interpretasi

Dari hasil clustering, didapatkan 5 kelompok:

Cluster	Karakteristik
1	Pendapatan sedang, pengeluaran sedang
2	Pendapatan rendah, pengeluaran rendah
3	Pendapatan tinggi, pengeluaran rendah
4	Pendapatan rendah, pengeluaran tinggi
5	Pendapatan tinggi, pengeluaran tinggi
💡 Insight Bisnis
Cluster 3 → Target promosi (punya uang tapi jarang belanja)
Cluster 5 → Customer loyal (harus dipertahankan)
Cluster 2 → Perlu strategi peningkatan daya beli
💾 Menyimpan Hasil
hasil_kmeans["CustomerID"] = dataset["CustomerID"]
hasil_kmeans.to_csv("hasil_clustering.csv", index=False)
