import gensim
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import pickle
from pprint import pprint
#funtion should be renamed extract_topics_from_transcripts
def extract_topics(processed_transcripts):
  # dictionary = gensim.corpora.Dictionary(processed_transcripts.values())
  # dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
  # bow_corpus = [dictionary.doc2bow(doc) for doc in processed_transcripts.values()]


  # tfidf = models.TfidfModel(bow_corpus)
  # corpus_tfidf = tfidf[bow_corpus]

  # lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

  # for idx, topic in lda_model.print_topics(-1):
  #   print('Topic: {} Word: {}'.format(idx, topic))
  # processed transcripts has the structure {video_id: [list of words]}
  extracted_topics = {}
  for transcript in processed_transcripts:
    try:
      bow_corpus, dictionary = build_corpus(processed_transcripts[transcript])
      model = build_lda_model([bow_corpus], dictionary)
      topics = model.print_topics(-1)
      extracted_topics[transcript] = topics
    except Exception as e:
      print(e)
      # print(model)
      # print(topics)
      extracted_topics[transcript] = "Error extracting topics"
  return extracted_topics

def build_corpus(processed_transcript):
  try:
    dictionary = gensim.corpora.Dictionary([processed_transcript])
      # dictionary.filter_extremes(no_below=1, no_above=0.1, keep_n=100000)
    bow_corpus = dictionary.doc2bow(processed_transcript)
  except Exception as e:
    print(e)
    return "Error building topics"
  return bow_corpus, dictionary

def build_tfidf_corpus(processed_transcript):
  try:
      dictionary = gensim.corpora.Dictionary([doc.split() for doc in processed_transcript])
      bow_corpus = [dictionary.doc2bow(doc.split()) for doc in processed_transcript]
      ## save reuse tfidfmodel
      tfidf_model = TfidfModel(bow_corpus)
      tfidf_model.save('tfidf_model')
      tfidf_corpus = tfidf_model[bow_corpus]
  except Exception as e:
    print(e)
    print(e.with_traceback)
    print(e.args)
    return e
  return dictionary, tfidf_corpus

#use tfidf model to transform 'processed_transcript' column in transcripts dataframe to tfidf vectors and make a new column in the dataframe
def tfidf_vectorize(transcript, tfIdf_model, dictionary):
  return tfIdf_model[dictionary.doc2bow(transcript)]

# define a function that does this: transcripts['topics'] = transcripts['tfidf_vector'].apply(lambda x: lda_model.get_document_topics(x, minimum_probability=0.0))
def get_topics_from_tfidf_vector(tfidf_vector, lda_model):
  return lda_model.get_document_topics(tfidf_vector, minimum_probability=0.0)

def build_lda_model(bow_corpus, dictionary, num_topics=4):
  try:
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10, workers=4)
    return lda_model
  except Exception as e:
    print(e)
    return "Error extracting topics"

def find_optimal_topics(corpus, dictionary, processed_transcripts):
    try:
    #range from 4 to 12
      num_topics_list = [4, 5, 6, 7, 8, 9, 10, 11, 12]
      optimal_num_topics = 0
      lda_models = []
      for num_topics in num_topics_list:
        # lda_model = gensim.models.LdaMulticore(
        #   corpus,
        #   num_topics=num_topics,
        #   id2word=dictionary,
        #   passes=10, 
        #   alpha='auto',
        #   eta='auto',
        #   workers=4)

        # use non multicore version for now
        lda_model = gensim.models.LdaModel(
          corpus,
          num_topics=num_topics,
          id2word=dictionary,
          passes=10,
          alpha='auto',
          eta='auto'
          )
        lda_models.append(lda_model)
    except Exception as e:
      print(e)
      print(e.with_traceback)

      return "Error creating lda models"
    

    try:
      # use list comprehension to create a list of values of processed transcripts
      text_list = [processed_transcripts[transcript] for transcript in processed_transcripts]
      # calculate coherence values for each model
      coherence_values = [CoherenceModel(model=model, texts=text_list, dictionary=dictionary, coherence='c_v').get_coherence() for model in lda_models]

      with open('coherence_values.txt', 'wb') as f:
        pickle.dump(coherence_values, f)

      # optimal number of topics found from the max coherence value
      optimal_num_topics = num_topics_list[coherence_values.index(max(coherence_values))]
    
    except Exception as e:
      print(e)
      print(e.with_traceback)
      print(e.args)
      return "Error finding optimal topics"
    
    return optimal_num_topics, coherence_values