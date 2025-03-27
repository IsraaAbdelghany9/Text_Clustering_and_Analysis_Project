from preprocessing import process_tweet
from feature_extraction import vectorize_text
from clustering import apply_kmeans, compute_linkage
from evaluation import compute_silhouette, purity_score
from visualization import plot_elbow_and_silhouette, tsne_cluster_visualization, hierarchical_clustering_dendrogram
from sklearn.datasets import fetch_20newsgroups

# Constants
K = 10

# Selected 3 categories from the 20 newsgroups dataset

categories = [
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

# Load data

News_df = fetch_20newsgroups(subset='all', categories=categories, 
                             shuffle=False, remove=('headers', 'footers', 'quotes'))


# # preprocess using the process_tweet function
# cleaned_docs = process_tweet(documents.data)

News_df_clean = []
for tweet in News_df.data:
    News_df_clean.append(process_tweet(tweet))
    
# combine back into a single string
cleaned_docs = [' '.join(doc) for doc in News_df_clean]

# Vectorize text
X, vectorizer = vectorize_text(cleaned_docs, ngram_range=(3,3))

# Apply clustering
model = apply_kmeans(X, n_clusters=K)

# Evaluate clustering
silhouette = compute_silhouette(X, model.labels_)
print(f"Silhouette Score: {silhouette}")
purity_score = purity_score(News_df.target, model.labels_)
print(f"Purity Score: {purity_score}")

# Visualize clusters
plot_elbow_and_silhouette(X)
tsne_cluster_visualization(X, model)
hierarchical_clustering_dendrogram(X)
