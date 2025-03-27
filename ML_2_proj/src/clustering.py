from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage

def apply_kmeans(X, n_clusters=5):
    """

    Applies K-Means clustering.
    
    Args:
        X (numpy.ndarray): Data to cluster.
        n_clusters (int): Number of clusters.

    Returns:
        KMeans: Fitted KMeans model.

    """
    model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=10, random_state=42)
    model.fit(X)
    return model

def compute_linkage(X, method='ward'):
    """
    
    Computes linkage matrix for hierarchical clustering.
    
    Args:
        X (scipy.sparse.csr.csr_matrix): Data to cluster.
        method (str): Linkage method.

    Returns:
        numpy.ndarray: Linkage matrix.
    
    """
    return linkage(X.toarray(), method)
