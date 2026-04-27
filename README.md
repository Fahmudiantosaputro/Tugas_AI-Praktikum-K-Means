# K-Means Menggunakan Python

## 1. Import Library

Import library dasar yang diperlukan yaitu pandas, numpy, matplotlib, dan seaborn.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 2. Import Dataset

Download terlebih dahulu data customer pada link sebelumnya dan lakukan import dataset.

```python
dataset = pd.read_csv("Mall_Customers.csv")
dataset.head()
```

---

## 3. Eksplorasi Data

### Ukuran Data

```python
X = dataset.iloc[:, 3:5]  # mengambil kolom "Annual Income" dan "Spending Score"
X.shape
```

Data terdiri dari 200 baris atau terdapat 200 customer. Karena kolomnya sudah direduksi, maka kolom sekarang hanya ada 2 yaitu “Annual Income” dan “Spending Score”.

---

### Cek Data Missing

```python
X.isnull().sum()
```

Tidak terdapat data yang missing.

---

### Ringkasan Data

```python
X.describe()
```

Terlihat bahwa rata-rata pendapatan tahunan dari customer adalah sebesar $60.560 dan rata-rata score pengeluarannya adalah sebesar 50,2.

---

## 4. Menentukan Jumlah Cluster (k)

Metode Elbow adalah salah satu cara dalam menentukan jumlah cluster (k) yang paling tepat untuk pemodelan K-Means. Jumlah cluster yang optimum ditunjukkan pada belokan atau siku dari grafik elbow.

```python
wcss = []  # wcss -> Within Cluster Sum of Squares

for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, random_state=14)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Visualisasi Metode Elbow
plt.plot(range(1, 15), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
```

Dari grafik Elbow, terlihat bahwa belokan atau siku terjadi pada jumlah cluster 5 sehingga inilah jumlah cluster yang akan digunakan pada metode K-Means.

---

## 5. Clustering Menggunakan K-Means

```python
kmeans = KMeans(n_clusters=5, random_state=14)
kmeans.fit(X)
```

---

## 6. Melihat Hasil Cluster

Hasil cluster untuk masing-masing data dapat dilihat dengan menggunakan atribut `labels_` atau menggunakan fungsi `fit_predict()`.

```python
kmeans.labels_

# Cara lain
# y_pred = kmeans.fit_predict(X)
```

Menggabungkan hasil cluster ke dalam dataframe:

```python
hasil_kmeans = X.copy()
hasil_kmeans["cluster"] = kmeans.labels_
hasil_kmeans.head()
```

---

## 7. Visualisasi Hasil Clustering

### Grafik Batang Frekuensi Cluster

```python
cluster_x = hasil_kmeans["cluster"].value_counts().index
cluster_y = hasil_kmeans["cluster"].value_counts().values

sns.barplot(x=cluster_x, y=cluster_y)
plt.title("Frekuensi Data pada Masing-Masing Cluster (KMeans)")
plt.xlabel("Cluster")
plt.ylabel("Frekuensi")
```

Terlihat bahwa mayoritas data masuk ke dalam cluster 0. Penulisan cluster 0 mengikuti index Python yang dimulai dari 0.

---

### Persiapan Scatter Plot

```python
ann_kmeans0 = hasil_kmeans[hasil_kmeans["cluster"] == 0].iloc[:, 0]
spend_kmeans0 = hasil_kmeans[hasil_kmeans["cluster"] == 0].iloc[:, 1]

ann_kmeans1 = hasil_kmeans[hasil_kmeans["cluster"] == 1].iloc[:, 0]
spend_kmeans1 = hasil_kmeans[hasil_kmeans["cluster"] == 1].iloc[:, 1]

ann_kmeans2 = hasil_kmeans[hasil_kmeans["cluster"] == 2].iloc[:, 0]
spend_kmeans2 = hasil_kmeans[hasil_kmeans["cluster"] == 2].iloc[:, 1]

ann_kmeans3 = hasil_kmeans[hasil_kmeans["cluster"] == 3].iloc[:, 0]
spend_kmeans3 = hasil_kmeans[hasil_kmeans["cluster"] == 3].iloc[:, 1]

ann_kmeans4 = hasil_kmeans[hasil_kmeans["cluster"] == 4].iloc[:, 0]
spend_kmeans4 = hasil_kmeans[hasil_kmeans["cluster"] == 4].iloc[:, 1]

centroid_cluster = kmeans.cluster_centers_
centroid_cluster
```

---

### Visualisasi Scatter Plot

```python
plt.scatter(ann_kmeans0, spend_kmeans0, s=80, c="blue", label="Cluster 1")
plt.scatter(ann_kmeans1, spend_kmeans1, s=80, c="orange", label="Cluster 2")
plt.scatter(ann_kmeans2, spend_kmeans2, s=80, c="green", label="Cluster 3")
plt.scatter(ann_kmeans3, spend_kmeans3, s=80, c="red", label="Cluster 4")
plt.scatter(ann_kmeans4, spend_kmeans4, s=80, c="magenta", label="Cluster 5")

# Centroid
plt.scatter(centroid_cluster[:, 0], centroid_cluster[:, 1], s=160, c="black", label="Centroids")

plt.title("Clusters of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
```

---

## 8. Interpretasi Hasil Clustering

Dengan melihat hasil scatter plot, dapat diketahui perilaku customer pada masing-masing cluster:

* Cluster 1 → Annual Income SEDANG, Spending Score SEDANG
* Cluster 2 → Annual Income RENDAH, Spending Score RENDAH
* Cluster 3 → Annual Income TINGGI, Spending Score RENDAH
* Cluster 4 → Annual Income RENDAH, Spending Score TINGGI
* Cluster 5 → Annual Income TINGGI, Spending Score TINGGI

Strategi:

* Cluster 3 → target promosi (income tinggi, belanja rendah)
* Cluster 5 → pelanggan loyal → beri reward atau promo khusus

---

## 9. Menggabungkan dengan Customer ID

```python
hasil_kmeans["CustomerID"] = dataset["CustomerID"]
hasil_kmeans.head()
```

---

## 10. Menyimpan Hasil Clustering

```python
hasil_kmeans.to_csv("Hasil Clustering Menggunakan K-Means.csv", index=False)
```

Link data set : https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
Menggunakan metode : K-Means Clustering
