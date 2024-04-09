import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error
from numpy import nanmean
import numpy as np
import math
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from tqdm import tqdm
from math import sqrt

def load_data(filepath):
    return pd.read_csv(filepath)

def align_emotions(df1, df2, key='video_id'):
    """Align the emotion columns of two dataframes."""
    emotion_columns = [col for col in df1.columns if col != key]
    df2_aligned = df2[[key] + emotion_columns]
    return df1, df2_aligned

def merge_data(df1, df2, key='video_id'):
    """Merge two dataframes on a common key."""
    return pd.merge(df1, df2, on=key, suffixes=('_genre', '_color'))


def calculate_correlation(emotions1, emotions2):
    """Calculate correlation coefficient between two emotion sets."""
    emotions1 = np.array(emotions1)
    emotions2 = np.array(emotions2)

    # Ensure both arrays have the same length
    if len(emotions1) != len(emotions2):
        return np.nan

    # Mean of the arrays
    mean1 = np.mean(emotions1)
    mean2 = np.mean(emotions2)

    # Calculate the Pearson correlation
    numerator = np.sum((emotions1 - mean1) * (emotions2 - mean2))
    denominator = np.sqrt(np.sum((emotions1 - mean1) ** 2) * np.sum((emotions2 - mean2) ** 2))

    if denominator == 0:
        return np.nan

    correlation = numerator / denominator
    return correlation


def calculate_mse(emotions1, emotions2):
    """Calculate mean squared error between two emotion sets."""
    return mean_squared_error(emotions1, emotions2)

def calculate_rmse(emotions1, emotions2):
    """Calculate mean squared error between two emotion sets."""
    return sqrt(mean_squared_error(emotions1, emotions2))

def calculate_kl_divergence(emotions1, emotions2):
    """Calculate KL Divergence between two emotion sets."""
    # Convert to numpy arrays and ensure type is float
    emotions1 = np.array(emotions1, dtype=float)
    emotions2 = np.array(emotions2, dtype=float)

    # Add a small constant to avoid zero probabilities
    epsilon = 1e-10
    emotions1 += epsilon
    emotions2 += epsilon

    # Re-normalize the distributions
    emotions1 /= np.sum(emotions1)
    emotions2 /= np.sum(emotions2)

    # Calculate KL Divergence
    return entropy(emotions1, emotions2)

def calculate_cosine_similarity(emotions1, emotions2):
    """Calculate Cosine Similarity between two emotion sets."""
    # Convert to numpy arrays
    emotions1 = np.array(emotions1, dtype=float)
    emotions2 = np.array(emotions2, dtype=float)

    # Add a small constant to avoid division by zero
    epsilon = 1e-10
    emotions1 += epsilon
    emotions2 += epsilon

    # Calculate norms
    norm1 = np.linalg.norm(emotions1)
    norm2 = np.linalg.norm(emotions2)

    # Calculate Cosine Similarity
    return np.dot(emotions1, emotions2) / (norm1 * norm2)



def calculate_js_divergence(emotions1, emotions2, epsilon=1e-10):
    # Adding epsilon to avoid log(0) and division by zero
    emotions1 = np.array(emotions1) + epsilon
    emotions2 = np.array(emotions2) + epsilon

    m = 0.5 * (emotions1 + emotions2)
    kl_div1 = calculate_kl_divergence(emotions1, m)
    kl_div2 = calculate_kl_divergence(emotions2, m)

    js_div = 0.5 * (kl_div1 + kl_div2)
    return js_div

def calculate_spearman_correlation(emotions1, emotions2):
    """Calculate Spearman Correlation"""
    # Convert to numpy arrays
    emotions1 = np.array(emotions1, dtype=float)
    emotions2 = np.array(emotions2, dtype=float)

    # Check for constant arrays
    if np.std(emotions1) == 0 or np.std(emotions2) == 0:
        return np.nan  # Return NaN if either array is constant

    # Calculate Spearman correlation
    correlation, _ = spearmanr(emotions1, emotions2)
    return correlation
def calculate_hellinger_distance(emotions1, emotions2):
    # Convert to numpy arrays
    emotions1 = np.array(emotions1, dtype=float)
    emotions2 = np.array(emotions2, dtype=float)

    # Calculate Hellinger distance
    return np.sqrt(np.sum((np.sqrt(emotions1) - np.sqrt(emotions2)) ** 2)) / np.sqrt(2)


def transform_kl_divergence(kl_values):
    exp_transformed = []
    sigmoid_transformed = []

    for kl in kl_values:
        # Exponential transformation
        exp_sim = math.exp(-kl)
        exp_transformed.append(exp_sim)

        # Sigmoid transformation
        sigmoid_sim = 1 / (1 + math.exp(kl))
        sigmoid_transformed.append(sigmoid_sim)

    return exp_transformed, sigmoid_transformed


def aggregate_results(df, emotion_columns):
    """Aggregate and display results of emotion comparison."""
    correlations, mses, rmses, kl_divergences, cosine_similarities, js_divergences, spearman_correlations, hellinger_distances = [], [], [], [], [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc = 'Calculating the Similarity'):
        emotions_genre = row[[col + '_genre' for col in emotion_columns]].values
        emotions_color = row[[col + '_color' for col in emotion_columns]].values

        correlations.append(calculate_correlation(emotions_color, emotions_genre))
        mses.append(calculate_mse(emotions_color, emotions_genre))
        rmses.append(calculate_rmse(emotions_color, emotions_genre))
        kl_divergences.append(calculate_kl_divergence(emotions_color, emotions_genre))
        cosine_similarities.append(calculate_cosine_similarity(emotions_color, emotions_genre))
        js_divergences.append(calculate_js_divergence(emotions_color, emotions_genre))
        spearman_correlations.append(calculate_spearman_correlation(emotions_color, emotions_genre))
        hellinger_distances.append(calculate_hellinger_distance(emotions_color, emotions_genre))

    return {
        'Average Pearson Correlation': nanmean(correlations),
        'Average MSE': np.mean(mses),
        'Average RMSE': np.mean(rmses),
        'Average KL Divergence': np.mean(kl_divergences),
        'Average Cosine Similarity': np.mean(cosine_similarities),
        'Jensen-Shannon Divergence': np.mean(js_divergences),
        "Spearman's Rank Correlation Coefficient": nanmean(spearman_correlations),
        'Hellinger Distance': np.mean(hellinger_distances)

    }

# Main execution
if __name__ == "__main__":
    df_genre_emotion = load_data('outputs/genre_emotion_weights.csv')
    df_color_emotion = load_data('outputs/movie_id_genre_video_mapped_emotion.csv')

    df_genre_emotion, df_color_emotion = align_emotions(df_genre_emotion, df_color_emotion)
    df_merged = merge_data(df_genre_emotion, df_color_emotion)

    emotion_columns = [col for col in df_genre_emotion.columns if col != 'video_id']

    results = aggregate_results(df_merged, emotion_columns)

    for metric, value in results.items():
        print(f"{metric}: {value}")
