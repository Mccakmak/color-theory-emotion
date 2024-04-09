from topic_modelling.text_processing import *
from topic_modelling.LDA_modelling import *
from sklearn.decomposition import PCA
from gensim import corpora, models, matutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def get_vectorized_topic(processed_transcripts):
    
    for transcript in processed_transcripts:
      try:
        dictionary , vectorized_corpus = build_tfidf_corpus(processed_transcripts[transcript])
        print(vectorized_corpus)
        # print porperties of the vectorized corpus
        matrix = np.transpose(matutils.corpus2dense(vectorized_corpus, len(dictionary)))
        
        pca = PCA(n_components=2)
        pca.fit(matrix)

        transformed_data = pca.transform(matrix)

        plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
        #show png image of the plot that will run for terminal
        plt.show()
        break

      except Exception as e:
        print("error in vectorizing topic")
        print(e.with_traceback)
        #error with line number
        print(e.args)
        break
        # print(topics)
    
def get_combined_vectorized_topic(processed_transcripts):
    #combine all the transcripts into list
    combined_transcripts = []
    for transcript in processed_transcripts:
      combined_transcripts += processed_transcripts[transcript]

    #build corpus
    dictionary , vectorized_corpus = build_tfidf_corpus(combined_transcripts)
    return vectorized_corpus, dictionary

#uses the dictionary and lda model as input
def get_top_words(model, dictionary):
   top_words = []

   for topic_id in range(model.num_topics):
     word_probabilities = model.get_topic_terms(topic_id, topn=10)
     topic_words = [dictionary[id] for id, _ in word_probabilities]
     top_words.append(topic_words)

   return top_words

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

def show_topic_visualization(model, corpus, dictionary):
  # Create visualization
  vis_data = gensimvis.prepare(model, corpus, dictionary)
  pyLDAvis.display(vis_data)
  pyLDAvis.save_html(vis_data, 'lda_visualization.html')

def load_Tfidf_model():
    # load tfidf model saved by .save() method
    tfIdf_model = TfidfModel.load('tfidf_model')
    return tfIdf_model

def show_topic_statistics(transcripts, lda_model, dictionary):
  # show distribution of strongest topics
  print(transcripts['strongest_topic'].value_counts())
  # show top words in each topic
  pprint(lda_model.print_topics())
  # show top words in each topic
  top_words = get_top_words(lda_model, dictionary)
  pprint(top_words)