from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import os

def save_plot(fig, filename):
    if not os.path.exists('../reports/figures'):
        os.makedirs('../reports/figures')
    fig.savefig(os.path.join('../reports/figures', filename))

def apply_kmeans(data, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=0)
    y_km = km.fit_predict(data)
    return km, y_km

def plot_clusters(data, y_km, km):
    fig, ax = plt.subplots()
    ax.scatter(data[y_km == 0, 0], data[y_km == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='Cluster 1')
    ax.scatter(data[y_km == 1, 0], data[y_km == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='Cluster 2')
    ax.scatter(data[y_km == 2, 0], data[y_km == 2, 1], s=50, c='lightblue', marker='v', edgecolor='black', label='Cluster 3')
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='Centroids')
    ax.legend()
    ax.grid()
    return fig

def apply_dbscan(data, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    y_db = db.fit_predict(data)
    return db, y_db

def plot_dbscan(data, y_db):
    fig, ax = plt.subplots()
    ax.scatter(data[y_db == 0, 0], data[y_db == 0, 1], c='lightblue', edgecolor='black', marker='o', s=40, label='Cluster 1')
    ax.scatter(data[y_db == 1, 0], data[y_db == 1, 1], c='red', edgecolor='black', marker='s', s=40, label='Cluster 2')
    ax.legend()
    return fig

def save_kmeans_plot(fig, filename='kmeans_clusters.png'):
    save_plot(fig, filename)

def save_dbscan_plot(fig, filename='dbscan_clusters.png'):
    save_plot(fig, filename)
