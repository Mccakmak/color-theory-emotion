import pandas as pd
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
import find_closest_color_emotion as fcce

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

def most_similar_emotion(main_emotions, emolex_df, emotion_words_dict, filename):
    all_mapped_emotions = {}
    for video_id, video_emotions in emotion_words_dict.items():
        mapped_emotions = {}
        for word, weight in video_emotions.items():
            scores = []
            for emotion in main_emotions:
                scores.append(find_similarity_score(emolex_df, word, emotion))
            # Find the most similar emotion using similarity score
            mapped_emotion = main_emotions[scores.index(max(scores))]
            mapped_emotions[mapped_emotion] = mapped_emotions.get(mapped_emotion, 0) + weight

        # Store all emotions
        all_mapped_emotions[video_id] = mapped_emotions

        #print(all_mapped_emotions)
        # SAVE

        # Create a DataFrame from the data
        df = pd.DataFrame(all_mapped_emotions.items(), columns=['video_id', 'emotions'])

        # Expand the emotions dictionary into separate columns
        emotions_df = pd.json_normalize(df['emotions'])

        # Concatenate the video_id column and expanded emotions columns
        result_df = pd.concat([df['video_id'], emotions_df], axis=1)

        # Save the DataFrame to a CSV file
        result_df.to_csv('outputs/'+filename + '_video_mapped_emotion.csv', index=False)


    return all_mapped_emotions
def combine_results(main_emotions, emolex_df, emotion_words_dict, filename):

    all_mapped_emotions = most_similar_emotion(main_emotions, emolex_df, emotion_words_dict, filename)

    combined_dict = {}
    for _, emotion_dict in all_mapped_emotions.items():
        for emotion, weight_str in emotion_dict.items():
            weight = float(weight_str)
            if emotion in combined_dict:
                combined_dict[emotion] += weight
            else:
                combined_dict[emotion] = weight

    # Calculate the total sum of weights
    total_weight = sum(combined_dict.values())

    # Normalize the weights
    normalized_result = {emotion: weight / total_weight for emotion, weight in combined_dict.items()}

    sorted_result = dict(sorted(normalized_result.items(), key=lambda item: item[0], reverse=True))

    # Convert the dictionary into a DataFrame, sort it by emotion in descending order
    df = pd.DataFrame(sorted(sorted_result.items(), key=lambda x: x[1], reverse=True), columns=['emotion', 'weight'])

    # Save the DataFrame to a CSV file
    df.to_csv('outputs/'+filename + '_combined_video_mapped_emotion.csv', index=False)
    return sorted_result

def load_nrc():
    # Load the lexicon
    emolex_df = pd.read_csv('NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
                            names=["word", "emotion", "association"], sep='\t')
    nrc_lexicon = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()

    # Load the NRC-VAD Lexicon
    #vad_df = pd.read_csv('NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt', names=["word", "valence", "arousal", "dominance"],sep='\t')

    # Merge the lexicons onto the existing DataFrame
    #vad_emolex_df = pd.merge(emolex_df, vad_df, on="word", how="left")

    return nrc_lexicon
def map_words_to_emotion(main_emotions, filename):

    # Load the lexicons
    nrc_lexicon = load_nrc()

    # Load words
    emotion_words_dict = fcce.retrieve_emotion_words_from_video(filename)

    #print(emotion_words_dict)

    # Mapped results
    combine_results(main_emotions, nrc_lexicon, emotion_words_dict, filename)

if __name__ == '__main__':

    # Mapped emotions
    #main_emotions = ['neutral', 'anger', 'joy', 'fear', 'surprise', 'sadness', 'disgust']

    main_emotions = ['anger', 'fear', 'anticipation', 'trust', 'surprise', 'sadness', 'joy', 'disgust']

    filename = 'movie_id_genre'

    map_words_to_emotion(main_emotions, filename)

