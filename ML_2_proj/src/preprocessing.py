import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

def process_tweet(tweet):
    """
    Cleans and tokenizes a tweet.
    
    Args:
        tweet (str): A tweet.

    Returns:
        list: A list of cleaned and tokenized words.
    
    """
    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')

    tweet = re.sub(r'\$\w*', '', tweet)  # Remove stock tickers
    tweet = re.sub(r'^RT[\s]+', '', tweet)  # Remove RT
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)  # Remove links
    tweet = re.sub(r'#', '', tweet)  # Remove hashtags
    tweet = re.sub(r'\d+', '', tweet)  # Remove numbers

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = [
        lemmatizer.lemmatize(word)
        for word in tweet_tokens
        if word not in stopwords_english and word not in string.punctuation
    ]

    return tweets_clean
