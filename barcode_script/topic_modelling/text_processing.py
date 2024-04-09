import gensim
import numpy as np
import nltk
import pandas as pd
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

np.random.seed(2023)

# lemmatize and stem preprocessing text, removing common words with gensim
def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
# returns a list of lemmatized and stemmed words
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
        # remove numbers
        if token.isnumeric():
            result.remove(token)
        # remove links
        if token.startswith('http'):
            result.remove(token)
    return result
# returns a dictionary of processed transcripts with video ids as keys

def preprocess_transcripts(transcripts):
    if not isinstance(transcripts, pd.DataFrame):
        transcripts = pd.DataFrame.from_dict(transcripts, orient='index', columns=['transcript']).reset_index().rename(columns={'index':'video_id'})

    processed_transcripts = {}
    for index, row in transcripts.iterrows():
      processed_transcripts[index] = preprocess(row['transcript'])

    return processed_transcripts