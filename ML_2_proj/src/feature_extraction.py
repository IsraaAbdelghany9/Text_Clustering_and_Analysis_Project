from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(text_data, ngram_range=(1,1)):
    """
    Converts text data into TF-IDF features.
    
    Args:
        text_data (list): A list of text data.
        ngram_range (tuple): The range of n-grams to consider.

    Returns:
        scipy.sparse.csr.csr_matrix: A matrix of TF-IDF features.
        TfidfVectorizer: A TfidfVectorizer object.
    
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range , stop_words='english', max_features=5000)
    return vectorizer.fit_transform(text_data), vectorizer
