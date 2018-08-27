# Firstly, please note that the performance of google word2vec is better on big datasets.
# In this example we are considering only 25000 training examples from the imdb dataset.
# Therefore, the performance is similar to the "bag of words" model.

# Importing libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
# BeautifulSoup is used to remove html tags from the text
from bs4 import BeautifulSoup
import re # For regular expressions

# Stopwords can be useful to undersand the semantics of the sentence.
# Therefore stopwords are not removed while creating the word2vec model.
# But they will be removed  while averaging feature vectors.
from nltk.corpus import stopwords
# Read data from files


#read the data from csv
reviews =  pd.read_csv('reviews.csv',error_bad_lines=False,sep='|')
print("Totalreviews :",len(reviews))
#get the length of the reviews
reviews['text length'] = reviews['text'].apply(len)
X = reviews['label']
Y = reviews['text']
reviews['label'] = np.where(reviews['label']=='positive' , 1, 0)

#change index of review to start from 1
reviews.index = reviews.index+1

#separate train data and test data
test_data = reviews[reviews.index % 5 == 0]
train_data = reviews[reviews.index % 5 != 0]
print("test data",len(test_data))
print("train data",len(train_data))
train_data_reviews = train_data['text']
train_data_label = train_data['label']
test_data_reviews = test_data['text']
test_data_label = test_data['label']

# This function converts a text to a sequence of words.
def review_wordlist(review_text, remove_stopwords=False):

    # 2. Removing non-letter.
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    # 3. Converting to lower case and splitting
    words = review_text.lower().split()
    # 4. Optionally remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)
# word2vec expects a list of lists.
# Using punkt tokenizer for better splitting of a paragraph into sentences.

import nltk.data
#nltk.download('popular')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# This function splits a review into sentences
def review_sentences(review_text, tokenizer, remove_stopwords=False):
    # 1. Using nltk tokenizer
    raw_sentences = tokenizer.tokenize(review_text.strip())
    sentences = []
    # 2. Loop for each sentence
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_wordlist(raw_sentence,
                                            remove_stopwords))

    # This returns the list of lists
    return sentences
sentences = []
print("Parsing sentences from training set")
for review in train_data_reviews:
    review = review_wordlist(review)
    review = str(review)
    sentences += review_sentences(review, tokenizer)


# Importing the built-in logging module
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Creating the model and setting values for the various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 40 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-3 # (0.001) Downsample setting for frequent words

# Initializing the train model
from gensim.models import word2vec
print("Training model....")
model = word2vec.Word2Vec(sentences,
                          workers=num_workers,
                          size=num_features,
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)

# To make the model memory efficient
model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


# Function to average all word vectors in a paragraph
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0

    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)

    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])

    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec
# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1

    return reviewFeatureVecs
# Calculating average feature vector for training set
clean_train_reviews = []
clean_test_reviews = []
for review in train_data_reviews:
    clean_train_reviews.append(review_wordlist(review, remove_stopwords=True))
for review in test_data_reviews:
    clean_test_reviews.append(review_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
testDataVecs = getAvgFeatureVecs()


# Fitting a random forest classifier to the training data
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)

print("Fitting random forest to training data....")
forest = forest.fit(trainDataVecs, train_data_label)
# Predicting the sentiment values for test data and saving the results in a csv file
result = forest.predict(testDataVecs)
print(np.mean(result == test_data_label))
