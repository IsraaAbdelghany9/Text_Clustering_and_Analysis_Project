from src.preprocessing import process_tweet
from src.feature_extraction import vectorize_text
from src.clustering import apply_kmeans, compute_linkage
from src.evaluation import compute_silhouette, purity_score
from src.visualization import plot_elbow_and_silhouette, tsne_cluster_visualization, hierarchical_clustering_dendrogram
from sklearn.datasets import fetch_20newsgroups

# Selected 3 categories from the 20 newsgroups dataset

categories = [
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]


# Load data

documents = fetch_20newsgroups(subset='all', categories=categories, 
                             shuffle=False, remove=('headers', 'footers', 'quotes'))


# Preprocess text
cleaned_docs = [" ".join(process_tweet(doc)) for doc in documents]

# Vectorize text
X, vectorizer = vectorize_text(cleaned_docs, ngram_range=(3,3))

# Apply clustering
model = apply_kmeans(X, n_clusters=3)

# Evaluate clustering
silhouette = compute_silhouette(X, model.labels_)
print(f"Silhouette Score: {silhouette}")

# Visualize clusters
plot_elbow_and_silhouette(X)
tsne_cluster_visualization(X, model)
hierarchical_clustering_dendrogram(X)
