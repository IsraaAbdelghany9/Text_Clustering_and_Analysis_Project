import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots

def plot_elbow_and_silhouette(X, k_range=range(2, 10)):
    """
    Plots Elbow Method and Silhouette Scores.
    
    Args:
        X (numpy.ndarray): Data.
        k_range (range): Range of k values.

    """
    inertia = []
    silhouette_scores = []

    for k in k_range:
        model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10, random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)
        if k > 1:  # Silhouette score is undefined for k=1
            score = silhouette_score(X, model.labels_)
            silhouette_scores.append(score)

    # Plot Elbow Method        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    plt.tight_layout()
    plt.show()




def tsne_cluster_visualization(X, model):
    """
    Visualizes clusters using t-SNE.
    
    Args:
        X (numpy.ndarray): Data.
        model (KMeans): Fitted KMeans model.
    
    """
    clusters = model.labels_.tolist()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X.toarray())

    plt.figure(figsize=(10, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters, cmap='tab10', alpha=0.6)
    plt.colorbar(label='Cluster')
    plt.title("t-SNE Visualization of KMeans Clusters")
    plt.show()



def hierarchical_clustering_dendrogram(X):
    """
    Plots a hierarchical clustering dendrogram.
    
    Args:
        X (scipy.sparse.csr.csr_matrix): Data.  
    
    """
    linkage_matrix = linkage(X.toarray(), 'ward')

    plt.figure(figsize=(15, 7))
    dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=10, color_threshold=0.7 * max(linkage_matrix[:, 2]), p=12, show_contracted=False)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()
