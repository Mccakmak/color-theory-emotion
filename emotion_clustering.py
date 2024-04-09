from scipy.spatial.distance import cosine, euclidean
import pickle
from gensim.models import KeyedVectors
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from collections import defaultdict


# It loads the NRC datas and prepares into proper format.
def prepare_data():
    # Load the lexicon
    emolex_df = pd.read_csv('NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',  names=["word", "emotion", "association"], sep='\t')
    emolex_df = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()

    # Load the NRC-VAD Lexicon
    vad_df = pd.read_csv('NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt',  names=["word", "valence", "arousal", "dominance"], sep='\t')

    #print(find_similarity_score(emolex_df, 'joy', 'happiness'))

    ### POSSIBLE FUTURE USAGE ###

    #Load the NRC-EIL Lexicon
    #eil_df = pd.read_csv('NRC-Emotion-Intensity-Lexicon/NRC-Emotion-Intensity-Lexicon-v1.txt',  names=["word", "emotion", "emotion_intensity"], sep='\t')
    #eil_df = eil_df.drop(columns=['emotion'])
    # Take the highest intensity scores
    #eil_df = eil_df.loc[eil_df.groupby('word')['emotion_intensity'].idxmax()]

    ### POSSIBLE FUTURE USAGE ###


    # Merge the lexicons onto the existing DataFrame
    emolex_words = pd.merge(emolex_df, vad_df, on="word", how="left")
    #emolex_words = pd.merge(emolex_words, eil_df, on="word", how="left")

    return emolex_words

# Finds the similarity score between 0 and 1 for two words
def find_similarity_score(emolex_df, input1, input2):
    word1 = emolex_df[emolex_df.word == input1]
    word2 = emolex_df[emolex_df.word == input2]

    vec1 = list(word1.iloc[0, 1:].values)
    vec2 = list(word2.iloc[0, 1:].values)

    if all(element == 0 for element in vec1) or all(element == 0 for element in vec2):
        # One of the vectors is a zero vector, return a predefined value (e.g., infinity).
        return 1 - euclidean(vec1, vec2)
    else:
        # Calculate the Cosine distance for non-zero vectors.
        return 1 - cosine(vec1, vec2)



# Retrieve our emotion keywords
def load_color_emotion_keywords():
    with open('color2emotion_weight_dict.pkl', 'rb') as f:
        color2emotion_weight_dict = pickle.load(f)

    # Using set to store unique emotions
    unique_emotions = set()

    # Iterate through the dictionaries and add emotions to the set
    for emotions_dict in color2emotion_weight_dict.values():
        unique_emotions.update(emotions_dict.keys())

    # Convert the set to a list and sort it alphabetically
    sorted_unique_emotions = sorted(unique_emotions)

    return sorted_unique_emotions

# Only include the word that is in lexicon
def filter_words(emolex_words):

    # Define your own words
    my_words = load_color_emotion_keywords()

    # Filter the lexicon to include only your words
    my_emolex_words = emolex_words[emolex_words['word'].isin(my_words)]

    removed_words = len(my_words) - len(my_emolex_words)

    if not removed_words == 0:
        print(removed_words, " words removed since they are not in lexicon")

    return my_emolex_words

def generate_dendrogram(words, data, max_d=2.5):
    # Linkage Matrix
    Z = linkage(data, method='ward')  # we use 'ward' as our linkage metric

    # Cluster assignment
    clusters = fcluster(Z, max_d, criterion='distance')

    # Group words by their cluster labels
    cluster_dict = defaultdict(list)
    for word, cluster_id in zip(words, clusters):
        cluster_dict[cluster_id].append(word)

    # Plotting Dendogram
    plt.figure(figsize=(15, 15))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    # Making Dendogram
    dendrogram(
        Z,
        leaf_rotation=0,
        leaf_font_size=12,
        color_threshold=max_d,  # This line will color the graph based on distance
        orientation='right',
        labels=words
    )
    plt.show()

    return cluster_dict


def compute_centroid(words, model):
    # Filter out words not in the model
    words = [word for word in words if word in model]
    # Return mean of word vectors
    return np.mean([model[word] for word in words], axis=0)


def closest_words_to_centroid(cluster, model, top_n=10):
    centroid = compute_centroid(cluster, model)
    # Return top N closest words to the centroid
    return model.similar_by_vector(centroid, topn=top_n)


def get_hypernyms(word):
    # Get synsets for the word
    synsets = wordnet.synsets(word)
    # If synsets are found, get hypernyms for the first synset (most common sense)
    if synsets:
        return synsets[0].hypernyms()
    return []


def get_hypernyms_level_2(word):
    synsets = wordnet.synsets(word)
    hypernyms_level_2 = []
    if synsets:
        hypernyms_1 = synsets[0].hypernyms()
        for hypernym in hypernyms_1:
            hypernyms_level_2.extend(hypernym.hypernyms())
    return hypernyms_level_2


if __name__ == '__main__':

    # Load the data
    emolex_words = prepare_data()

    # Filter the words
    df = filter_words(emolex_words)
    words = df['word'].tolist()
    df = df.drop(columns=['word'])
    clusters = generate_dendrogram(words, df)

    print(clusters)

    # Avoid extremely general words
    exclude_list = ['emotion', 'feeling', 'feel', 'state', 'emotional_state',
                    'condition', 'attitude', 'attribute', 'situation', 'psychological_state',
                    'cognitive_state']

    # Load Word2Vec model (this may take some time)
    model_path = "glove/glove.6B.300d.word2vec"
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    for cluster_id, words in clusters.items():
        print(f"Cluster {cluster_id}")
        print()
        print("Words: ", words)
        print()
        close_words = closest_words_to_centroid(words, word2vec_model)
        print("Closest words to centroid: ", close_words)
        print()
        suggested_hypernyms_level_1 = set()
        suggested_hypernyms_level_2 = set()
        for word, _ in close_words:
            # Level 1
            hypernyms_1 = get_hypernyms(word)
            for hypernym in hypernyms_1:
                hypernym_name = hypernym.lemmas()[0].name()
                if hypernym_name not in exclude_list:
                    suggested_hypernyms_level_1.add(hypernym_name)

            # Level 2
            hypernyms_2 = get_hypernyms_level_2(word)
            for hypernym in hypernyms_2:
                hypernym_name = hypernym.lemmas()[0].name()
                if hypernym_name not in exclude_list:
                    suggested_hypernyms_level_2.add(hypernym_name)

        print("Hypernyms Level 1:", suggested_hypernyms_level_1)
        print("Hypernyms Level 2:", suggested_hypernyms_level_2)
        print("-----------------------------")