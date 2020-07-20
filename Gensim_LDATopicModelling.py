#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Breaks the large raw tweet data file into manageable portions

'''


import pandas as pd

#setting file siz;e to 10000 tweets
chunk_size = 10000
batch_no = 1

print("Beginning to parse tweets...")

#Adjust path name per file, this is the small test file I am running currently
for chunk in pd.read_csv("./tweets_01-04.csv", chunksize = chunk_size, error_bad_lines=False):
        chunk.to_csv("covid_data" + str(batch_no) + ".csv", index = False)
        batch_no += 1

print("Finished parsing tweets.")


# In[ ]:


get_ipython().system('pip install gensim')
get_ipython().system('pip install nltk')
get_ipython().system('pip install numpy')
get_ipython().system('pip install csv')
get_ipython().system('pip install glob')
get_ipython().system('pip install pyLDAvis')
get_ipython().system('pip install spacy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install logging')
get_ipython().system('pip install warnings')
get_ipython().system('pip install wordcloud')


# In[ ]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
import csv
import glob
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import spacy
import pyLDAvis.gensim
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[ ]:


def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in ["https", "rtrt"] and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[ ]:


'''
 creates corpus for topic modelling 
'''

#Correct Dates start at file 700

sentimentDict = {}
documents = []
path = "./covid_data*"

num_files = 0

print("Beginning to create corpus...")
#I put the smaller chunked tweet files in to a directory called revisedCovidData
for filename in glob.glob(path):
    if (num_files)%5 == 0:
        #Iterate through this directory, reading each file
        with open(filename, 'r', encoding="utf-8") as rawTweets:
            #open as CSV iterator
            readCSV = csv.reader(rawTweets)
            next(readCSV)
            #Iterate through individual tweets
            for line in readCSV:
                #if line[9] != "Null" or "us_state":
                    #calls text of each tweet as a TextBlob object
                text = line[1]
                    #line[9] = state; if this state is already in the dictionary, the sentiment gets averaged
                documents.append(text)
    num_files += 1
    #break    
        
print(len(documents))
print(documents[-21])


# In[ ]:


doc_sample = documents[4310]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


# In[ ]:





# In[ ]:


def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in ["rtrt", "https"] and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[ ]:


doc_sample = documents[370]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


# In[ ]:


print("working")
preprocessed_docs = []
print("processing the documents")
for tweet in documents:
    preprocessed_docs.append(preprocess(tweet))


# In[ ]:



print("Creating dictionary")
dictionary = gensim.corpora.Dictionary(preprocessed_docs)

count = 0
for k, v in dictionary.iteritems():
    print(k,v)


# In[ ]:


#filters out any token in 15 or fewer, more than half, and only the 100,000 most common
dictionary.filter_extremes(no_below = 15, no_above = 0.5, keep_n = 100000)


# In[ ]:


print(preprocessed_docs[3])
bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]


# In[ ]:


bow_doc_5 = bow_corpus[5]
for i in range(len(bow_doc_5)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_5[i][0], 
                                               dictionary[bow_doc_5[i][0]], 
bow_doc_5[i][1]))


# In[ ]:





# In[ ]:


from gensim import corpora, models
id2word = corpora.Dictionary(preprocessed_docs)
texts = preprocessed_docs
corpus = [id2word.doc2bow(text) for text in texts]
print(corpus[:1])


# In[ ]:



'''
Build LDA model
'''
lda_model = gensim.models.ldamodel.LdaModel(corpus = bow_corpus,
                                           id2word=id2word,
                                           num_topics = 20,
                                           random_state = 100, 
                                           update_every=1,
                                           chunksize = 100,
                                           passes = 10,
                                           alpha='auto',
                                           per_word_topics = True)


# In[ ]:


pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[ ]:


lda_model.save('lda.model')


# In[ ]:


#measure how good the model is. Lower is better
print('\nPerplexity: ', lda_model.log_perplexity(corpus))

#compute coherence score
coherence_model_lda = CoherenceModel(model=lda_model, texts = preprocessed_docs, dictionary = id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:


# Compute Coherence Score using UMass
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence="u_mass")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[ ]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[ ]:


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=texts, start=2, limit=40, step=6)
# Show graph
import matplotlib.pyplot as plt
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[ ]:


# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
for text in texts:
    for word in text:
        long_string = long_string + ',' + word
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()
wordcloud.to_file("img/wordcloud.png")


# In[ ]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




