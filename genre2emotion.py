from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ast

def read_dataset(file_path):
    return pd.read_csv(file_path)

def load_nrc():
    # Load the lexicon
    emolex_df = pd.read_csv('NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
                            names=["word", "emotion", "association"], sep='\t')
    nrc_lexicon = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
    return nrc_lexicon


# Function to find a similar word in the NRC Emotion Lexicon
def find_similar_word(word, nrc_lexicon):
    # Mapping for known compound words or abbreviations
    known_mappings = {
        'sci-fi': ['science', 'fiction'],
        'family':['household'],
        'reality-tv':['reality']
    }

    # Check if the word is a known compound word or abbreviation
    if word in known_mappings:
        for mapped_word in known_mappings[word]:
            if mapped_word in nrc_lexicon['word'].values:
                return mapped_word

    # Original checks for the exact word and similar words
    if word in nrc_lexicon['word'].values:
        return word

    # List of common suffixes to try
    suffixes = ['er', 'or', 'ing', 'ly', 'ed', 's', 'es']
    for suffix in suffixes:
        if word.endswith(suffix):
            # Try the word without the suffix
            root_word = word[:-len(suffix)]
            if root_word in nrc_lexicon['word'].values:
                return root_word

    # If no match is found, return the original word (or None if preferred)
    return word


def map_genres_to_emotions(genres, nrc_lexicon):
    emotions = defaultdict(int)
    excluded_emotions = ['positive', 'negative']  # Exclude these emotions
    for genre in genres:
        similar_genre = find_similar_word(genre.lower(), nrc_lexicon)
        genre_emotions = nrc_lexicon[nrc_lexicon['word'] == similar_genre]
        for emotion in genre_emotions.columns[1:]:  # Excluding 'word' column
            if emotion not in excluded_emotions:
                emotions[emotion] += int(genre_emotions.iloc[0][emotion])
    return emotions

def normalize_emotion_scores(emotions):
    total = sum(emotions.values())
    return {emotion: score / total if total > 0 else 0 for emotion, score in emotions.items()}


def process_dataset(df, nrc_lexicon):
    extracted_data = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], unit='video', desc='Mapping genres to emotions'):
        video_id = row['video_id']
        genres = ast.literal_eval(row['genres'])

        emotions = map_genres_to_emotions(genres, nrc_lexicon)
        normalized_emotions = normalize_emotion_scores(emotions)
        normalized_emotions['video_id'] = video_id
        extracted_data.append(normalized_emotions)

    # Creating a DataFrame with each emotion as a separate column
    df_weight = pd.DataFrame(extracted_data)
    df_weight = df_weight.fillna(0)  # Fill missing values with 0
    df_weight = df_weight.set_index('video_id').reset_index()  # Set YouTube ID as the first column

    df_weight.to_csv("genre_emotion_weights.csv", index=False)

    return df_weight

# Preprocessing for plotting
def preprocess_data(data):
    data['genres'] = data['genres'].apply(ast.literal_eval)
    return data

# Counting the Genres
def analyze_genres(data):
    all_genres = [genre for sublist in data['genres'] for genre in sublist]
    genre_counts = Counter(all_genres)
    genre_distribution = pd.DataFrame.from_dict(genre_counts, orient='index', columns=['Count'])
    return genre_distribution.sort_values(by='Count', ascending=False)

# Visualization
def visualize_top_genres(genre_distribution, file_name, top_n=10):
    sns.set(style="whitegrid")
    top_genres = genre_distribution.head(top_n).reset_index()
    top_genres.rename(columns={'index': 'Genre'}, inplace=True)

    plt.figure(figsize=(12, 8))
    genre_plot = sns.barplot(x='Count', y='Genre', data=top_genres, palette='gray')
    #plt.title(f'Top {top_n} Movie Genres by Occurrence', fontsize=16)
    plt.xlabel('Number of Occurrences', fontsize=18)
    plt.ylabel('Genre', fontsize=18)

    # Change the font size of x and y axis values (tick labels)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    # Save the figure to a PDF file
    plt.savefig(file_name, format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    # Load the dataset
    dataset_path = 'inputs/movie_id_genre.csv'
    df = read_dataset(dataset_path)

    # Load the NRC Emotion Lexicon
    nrc_lexicon = load_nrc()

    # Process the dataset
    emotion_df = process_dataset(df, nrc_lexicon)


    # Visualization of the Genres
    data = read_dataset(dataset_path)
    processed_data = preprocess_data(data)
    genre_distribution = analyze_genres(processed_data)
    visualize_top_genres(genre_distribution, file_name="plots/genre_distrubition.pdf", top_n=10)