from preprocessing import process_tweet
from feature_extraction import vectorize_text
from clustering import apply_kmeans, compute_linkage
from evaluation import compute_silhouette, purity_score
from visualization import plot_elbow_and_silhouette, tsne_cluster_visualization, hierarchical_clustering_dendrogram
from sklearn.datasets import fetch_20newsgroups
import joblib

# Constants
K = 4

# Selected 3 categories from the 20 newsgroups dataset
categories = [
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

# Load data
News_df = fetch_20newsgroups(subset='all', categories=categories, 
                             shuffle=False, remove=('headers', 'footers', 'quotes'))


News_df_clean = []
for tweet in News_df.data:
    News_df_clean.append(process_tweet(tweet))
    
# combine back into a single string
cleaned_docs = [' '.join(doc) for doc in News_df_clean]

# Vectorize text
X, vectorizer = vectorize_text(cleaned_docs, ngram_range=(3,3))

# Apply clustering
model = apply_kmeans(X, n_clusters=K)

# Compute linkage
linkage_matrix = compute_linkage(X, method='ward')

# Evaluate clustering
silhouette = compute_silhouette(X, model.labels_)
print(f"Silhouette Score: {silhouette}")
purity_score = purity_score(News_df.target, model.labels_)
print(f"Purity Score: {purity_score}")

# Visualize clusters
plot_elbow_and_silhouette(X)
tsne_cluster_visualization(X, model)
hierarchical_clustering_dendrogram(X, linkage_matrix)


# Save model and the results 
# K_means model and vectorizer
joblib.dump(model, "ML_2_proj/results/kmeans_model.joblib")
joblib.dump(vectorizer, "ML_2_proj/results/vectorizer.joblib")

# clustering results
joblib.dump(model.labels_, "ML_2_proj/results/labels.joblib")
joblib.dump(linkage_matrix, "ML_2_proj/results/linkage_matrix.joblib")

# evaluation results
joblib.dump(silhouette, "ML_2_proj/results/silhouette.joblib")
joblib.dump(purity_score, "ML_2_proj/results/purity_score.joblib")




