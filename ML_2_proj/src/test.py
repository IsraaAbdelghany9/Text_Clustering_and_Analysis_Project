from sklearn.datasets import make_blobs
from visualization import plot_elbow_and_silhouette, tsne_cluster_visualization
from sklearn.cluster import KMeans

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Call the function
plot_elbow_and_silhouette(X)


# Generate sample data
model = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=10, random_state=42)
model.fit(X)

# Call the function
tsne_cluster_visualization(X, model)