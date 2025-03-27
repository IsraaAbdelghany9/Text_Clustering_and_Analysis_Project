import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    if not isinstance(tweet, str):
        raise ValueError(f"Expected a string, but got {type(tweet)}")

    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')

    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks    
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags only removing the hash # sign 
    tweet = re.sub(r'#', '', tweet)
    # remove numbers
    tweet = re.sub(r'\d+', '', tweet)


    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    # Clean tokens: remove stopwords and punctuation, then lemmatize
    tweets_clean = [
        lemmatizer.lemmatize(word)
        for word in tweet_tokens
        if word not in stopwords_english and word not in string.punctuation
    ]

    return tweets_clean

