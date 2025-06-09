# can inherent the functions and classes, not libraries.
# input the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
# cluster on one-hot encoded user demographic data, K-means, DBSCAN, Agglomerative Clustering
# Two steps: first we want to detect the outliers, and outliers should be one cluster
# Based on not outliers, I used the k-means methods to do the clustering process
rst = pd.read_excel('../data/ERR.xlsx', sheet_name='Restaurants')
rev = pd.read_csv('../data/cleaned_err.csv')

# combine the rst and rev file, and if not in the rst file, we dropped the reviews directly.
rev = pd.merge(rst, rev, on = 'Restaurant Name', how = 'right')
unique_count = rev.groupby('Restaurant Name')['Rating'].count()
missing_vals = ['La Principal', 'Todoroki Sushi','World Market',"Clare's Korner"]
rev = rev[~rev['Restaurant Name'].isin(missing_vals)]
rst = rev.copy()

rev_c = rev.drop_duplicates(subset = ['Reviewer Name'])
categorical_col = ['Marital Status', 'Has Children?', 'Preferred Mode of Transport',
                   'Northwestern Student?', 'Average Amount Spent']
numerical_col = ['Weight (lb)', 'Height (cm)', 'Age']
cols = categorical_col + numerical_col
rev2 = rev_c.loc[:,cols]
# one-hot for categorical data
rev2.loc[rev2['Marital Status'].str.startswith('S'), 'Marital Status'] = 'single'
rev2.loc[rev2['Marital Status'].str.startswith('W'), 'Marital Status'] = 'widow'
# 
rev2 = pd.get_dummies(rev2, columns=categorical_col, drop_first=True)
print(rev2.columns)
# standardize the numerical data
X = rev2.loc[:, numerical_col]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rev2.loc[:,numerical_col] = X_scaled
# use pca to reduce the dimension so we can display in the plot.
pca = PCA(n_components=2)
X_2d = pca.fit_transform(rev2)
cluster = [3,5,6,4]
# which means this is not a good method, cause it has a very small silhouette score.
for i in cluster:
    agglo = AgglomerativeClustering(n_clusters = i)
    y_pred = agglo.fit_predict(X)
    score = silhouette_score(X, y_pred)
    print(f"Silhouette Score: {score:.3f}")

agglo = AgglomerativeClustering(n_clusters = 4)
y_pred = agglo.fit_predict(X_2d)
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='viridis', s=50)
plt.title('Agglomerative Clustering (2D PCA Projection)')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.grid(True)
plt.show()
# try dbscan method: best result is when eps = 0.5, and min_samples = 3.
# and the silhouette-score is equal to 0.491
epses = [0.1, 0.2, 0.3, 0.4, 0.5]
for i in epses:
    db = DBSCAN(eps=i, min_samples=3).fit(X_2d)
    db_ypred = db.labels_
    score = silhouette_score(X_2d, db_ypred)
    print(f"DBSCAN methods Silhouette Score: {score:.3f}")
db = DBSCAN(eps=0.3, min_samples=5).fit(X_2d)
db_ypred = db.labels_
db_outliers = db_ypred == -1  
rev3 = rev2[~db_outliers].copy()
rev2['Clusters'] = np.where(db_outliers, 4, 0)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=db_ypred, cmap='viridis', s=50)
plt.title("DBSCAN Clustering (2D PCA)")
plt.xlabel("PCA Feature 1")
plt.ylabel("PCA Feature 2")
plt.grid(True)
plt.show()
# try the k-means method: 4 looks good.
for i in cluster:
    kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
    km_ypred =  kmeans.fit_predict(X_2d)
    score = silhouette_score(X_2d, km_ypred)
    print(f"Kmeans methods Silhouette Score: {score:.3f}")
pca = PCA(n_components=2)
X_new = pca.fit_transform(rev3)
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init='auto')
km_labels = kmeans_final.fit_predict(X_new)
rev3['Clusters'] = km_labels
rev2.update(rev3)

rev2['Rating'] = rev_c['Rating'].values
rev2['Restaurant Name'] = rev_c['Restaurant Name'].values

rev_scores = rev2.groupby('Clusters')['Rating'].mean().reset_index()
rev_scores.columns = ['Clusters', 'avg_Scores']
print(rev_scores)
pd.set_option('display.max_columns', None)
features = ['Weight (lb)', 'Height (cm)', 'Age', 'Has Children?_Yes', 'Preferred Mode of Transport_On Foot', 'Preferred Mode of Transport_Public Transit',
            'Northwestern Student?_Yes', 'Average Amount Spent_Low']
cluster_summary = rev2.groupby('Clusters')[features].mean()
print(cluster_summary)
# print('min:', rev_scores.min(), 'max:',rev_scores.max())
rev2 = pd.merge(rev2, rev_scores, on='Clusters')
rev2.to_csv('../data/match.csv')
rst.to_csv('../data/cleaned_err.csv')
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=km_ypred, cmap='viridis')
plt.title("KMeans Clustering Example")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# We decided to use the K-means methods, because high silhouette-score, and easy to understand.

