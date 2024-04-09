import pandas as pd
from tqdm import tqdm

# Function to read the dataset
def read_dataset(file_path):
    return pd.read_json(file_path)

# Function to extract YouTube IDs and genres and save as a CSV file
def save_youtube_id_and_genres_as_csv(df, output_file):
    extracted_data = []

    for entry in tqdm(df['trailers12k'], total=len(df['trailers12k']), unit='video', desc='Extracting YouTube IDs and genres'):
        # Check if 'youtube' and 'trailers' keys exist and extract YouTube IDs
        if 'youtube' in entry and 'trailers' in entry['youtube']:
            youtube_ids = [trailer['id'] for trailer in entry['youtube']['trailers']]

            # Check if 'genres' key exists in the 'imdb' section and extract genres
            genres = entry['imdb']['genres'] if 'genres' in entry['imdb'] else []

            for youtube_id in youtube_ids:
                extracted_data.append({'youtube_id': youtube_id, 'genres': genres})

    # Convert the list to a DataFrame and save as CSV
    extracted_df = pd.DataFrame(extracted_data)
    extracted_df.to_csv(output_file, index=False)

# Main execution
if __name__ == "__main__":
    # Load the dataset
    dataset_path = 'input/metadata.json'
    df = read_dataset(dataset_path)

    # Save YouTube IDs and genres as a CSV file
    output_csv_path = 'input/movie_id_genre.csv'
    save_youtube_id_and_genres_as_csv(df, output_csv_path)