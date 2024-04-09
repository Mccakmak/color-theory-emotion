from discover.discover import *
from utils.utils import *
import numpy as np
import pandas as pd
from utils.barcode import *
from utils.transcription import *
from topic_modelling.text_processing import *
from topic_modelling.LDA_modelling import *
from scripts.demo_all_functions import *
from topic_modelling.topic_similarty import *
from pprint import pprint

import pickle

def main():
  # videos = discover_videos('w2UV6HIpAog')
  # # save videos to csv keeping all columns, name csv videos+timestamp
  # save_video_to_csv(videos)

  # # get video ids from videos
  # video_ids = [video['id'] for video in videos]
  # # get transcripts for videos
  # transcripts = get_transcripts(video_ids)
  # # save transcripts to csv
  # pd.DataFrame.from_dict(transcripts, orient='index', columns=['transcript']).to_csv('transcripts.csv')
  transcripts = pd.read_csv('transcripts.csv', index_col=0)
  try:
    transcripts['transcript'] = transcripts['transcript'].str.replace(r'\n', ' ')
  except Exception as e:
    print(e)
    #try to remove the \n from the transcript a different way
    transcripts['transcript'] = transcripts['transcript'].apply(lambda x: x.replace(r'\n', ' '))


  processed_transcripts = preprocess_transcripts(transcripts)

  #print keys of dictionary
  # add values from the dictionary preprocessed transcripts to a new column in the transcripts dataframe
  transcripts['processed_transcript'] = transcripts.index.map(processed_transcripts)
  # use processed_transcripts to build 
  vectorized_transcripts, dictionary = get_combined_vectorized_topic(processed_transcripts)
  
  print(vectorized_transcripts)
  
  # # optimal_topics, coherence_values = find_optimal_topics(vectorized_transcripts, dictionary, preprocessed_transcripts)
  optimal_topics = 6

  # lda_model = build_lda_model(vectorized_transcripts, dictionary, optimal_topics)
  #save ldamodel, vectorized_transcripts, dictionary to pickle file

  # with open('ldamodel.pickle', 'wb') as handle:
  #   pickle.dump(lda_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # with open('vectorized_transcripts.pickle', 'wb') as handle:
  #   pickle.dump(vectorized_transcripts, handle, protocol=pickle.HIGHEST_PROTOCOL)
  # with open('dictionary.pickle', 'wb') as handle:
  #   pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

  #load ldamodel, vectorized_transcripts, dictionary from pickle file
  with open('ldamodel.pickle', 'rb') as handle:
    lda_model = pickle.load(handle)
  with open('vectorized_transcripts.pickle', 'rb') as handle:
    vectorized_transcripts = pickle.load(handle)
  with open('dictionary.pickle', 'rb') as handle:
    dictionary = pickle.load(handle)
  
  # pprint(lda_model.print_topics())
  # top_words = get_top_words(lda_model, dictionary)
  # pprint(top_words)
  # show_topic_visualization(lda_model, vectorized_transcripts, dictionary)

  # use trained LDA model to label videos

  tfIdf_model = load_Tfidf_model(); 


  # transcripts['tfidf_vector'] = transcripts['processed_transcript'].apply(lambda x: tfIdf_model[dictionary.doc2bow(x)])
  transcripts['tfidf_vector'] = transcripts['processed_transcript'].apply(lambda x: tfidf_vectorize(x, tfIdf_model, dictionary))
  # use the trained LDA model to label videos in a new column in the transcripts dataframe
  transcripts['topics'] = transcripts['tfidf_vector'].apply(lambda x: get_topics_from_tfidf_vector(x, lda_model))
  # Assign a topic to each video based on the topic with the highest probability in a new column called strongest_topic
  print(transcripts['topics'].head())
  # transcripts['topics'] has the form [(topic, probability), (topic, probability), ...]
  # we want to get the topic with the highest probability
  transcripts['strongest_topic'] = transcripts['topics'].apply(lambda x: max(x, key=lambda item: item[1])[0])

   #save transcripts dataframe to csv
  transcripts.to_csv('transcripts_transformed.csv')

  # show some statistics about the topics based on transcripts dataframe such as strongest topic distribution and what words are in each topic
  show_topic_statistics(transcripts, lda_model, dictionary)

if __name__ == "__main__":
  main()