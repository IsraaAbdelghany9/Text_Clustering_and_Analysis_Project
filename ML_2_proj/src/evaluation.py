import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import metrics



def compute_silhouette(X, labels):
    """
    Computes Silhouette Score for clustering.
    
    Args:
        X (numpy.ndarray): Data.
        labels (numpy.ndarray): Cluster labels.

    Returns:    
        float: Silhouette Score.

    """
    return silhouette_score(X, labels)



def purity_score(y_true, y_pred):
    """
    Computes Purity Score.
    
    Args:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: Purity Score.
    
    """
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
