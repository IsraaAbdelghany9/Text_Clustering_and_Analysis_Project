import re
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE 
import numpy as np 
from sklearn import metrics

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks    
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            lemma_word = lemmatizer.lemmatize(word)  # lemmatizing word
            tweets_clean.append(lemma_word)

    return tweets_clean


def plot_elbow_and_silhouette(X, k_range=range(2, 10)):
    inertia = []
    silhouette_scores = []
    K_range = k_range  # Checking for k from 2 to 10

    for k in K_range:
        model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=10, random_state=42)
        model.fit(X)
    
        inertia.append(model.inertia_)  # Sum of squared distances to closest cluster center
        silhouette_scores.append(silhouette_score(X, model.labels_))  # Measure of cluster quality

    # Plot Elbow Method
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertia, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    plt.show()


def tsne_cluster_visualization(X, model):
    clusters = model.labels_.tolist()

    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X.toarray())

    # Plot t-SNE clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters, cmap='tab10', alpha=0.6)
    plt.colorbar(label='Cluster')
    plt.title("t-SNE Visualization of KMeans Clusters")
    plt.show()


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
