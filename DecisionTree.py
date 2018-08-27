import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models import word2vec



#read the data from csv
reviews =  pd.read_csv('reviews.csv',error_bad_lines=False,sep='|')
print("Totalreviews :",len(reviews))
#get the length of the reviews
reviews['text length'] = reviews['text'].apply(len)

#convert labels positive to 1 and negative to 0
reviews['label'] = np.where(reviews['label']=='positive' , 1, 0)


X = reviews['label']
Y = reviews['text']

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

#generate CSV for visualisation
'''train_data_reviews.to_csv('train_data_reviews.csv', sep='|', encoding='utf-8')
train_data_label.to_csv('train_data_label.csv', sep='|', encoding='utf-8')
test_data_reviews.to_csv('test_data_reviews.csv', sep='|', encoding='utf-8')
test_data_label.to_csv('test_data_label.csv', sep='|', encoding='utf-8')



#Convert a raw review to a cleaned review

countVect = CountVectorizer()
X_train_countVect = countVect.fit_transform(train_data_reviews)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_countVect)

X_new_counts = countVect.transform(test_data_reviews)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)'''

'''print("Decision Tree :")

dtc = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                               max_depth=5, min_samples_leaf=10)
dtc = dtc.fit(X_train_tfidf,train_data_label)

predictedDT = dtc.predict(X_new_tfidf)
print(np.mean(predictedDT == test_data_label))
accuracy = accuracy_score(predictedDT,test_data_label)
print(accuracy)'''
#modelEvaluation(predictedDT)

#using grid search and pipeline for logistic regression

def cleanText(raw_text, remove_stopwords=False, stemming=False, split_text=False ):
    '''
    Convert a raw review to a cleaned review
    '''

    letters_only = re.sub("[^a-zA-Z]", " ", raw_text)  # remove non-character
    words = letters_only.lower().split() # convert to lower case

    if remove_stopwords: # remove stopword
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    if stemming==True: # stemming
#         stemmer = PorterStemmer()
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]

    if split_text==True:  # split text
        return (words)

    return( " ".join(words))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def parseSent(reviewtext, tokenizer, remove_stopwords=False):
    '''
    Parse text into sentences
    '''
    raw_sentences = tokenizer.tokenize(reviewtext.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(cleanText(raw_sentence, remove_stopwords, split_text=True))
    return sentences


# Parse each review in the training set into sentences
'''sentences = []
for reviewtext in train_data_reviews:
    sentences += parseSent(reviewtext, tokenizer)'''
#word2vec and RNN

num_features = 300  #embedding dimension
min_word_count = 10
num_workers = 4
context = 10
downsampling = 1e-3

print("Training Word2Vec model ...\n")
w2v = word2vec.Word2Vec.load('Word2vectorfile')
#word2vec.Word2Vec(X_train_cleaned, workers=num_workers, size=num_features, min_count = min_word_count,  window = context, sample = downsampling)

def makeFeatureVec(reviewtext, model, num_features):
    
    #Transform a review to a feature vector by averaging feature vectors of words
    #appeared in that review and in the volcabulary list created
    
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word) #index2word is the volcabulary list of the Word2Vec model
    isZeroVec = True
    for word in reviewtext:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
            isZeroVec = False
    if isZeroVec == False:
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviewtext, model, num_features):
    
    #Transform all reviews to feature vectors using makeFeatureVec()
    
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviewtext),num_features),dtype="float32")
    for review in reviewtext:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
        counter = counter + 1
    return reviewFeatureVecs

X_train_cleaned = []
for review in train_data_reviews:
    X_train_cleaned.append(cleanText(review, remove_stopwords=True, split_text=True))

trainVector = getAvgFeatureVecs(X_train_cleaned, w2v, num_features)
print("Training set : %d feature vectors with %d dimensions" %trainVector.shape)


# Get feature vectors for validation set
X_test_cleaned = []
for review in test_data_reviews:
    X_test_cleaned.append(cleanText(review, remove_stopwords=True, split_text=True))
testVector = getAvgFeatureVecs(X_test_cleaned, w2v, num_features)
print("Validation set : %d feature vectors with %d dimensions" %testVector.shape)


# Random Forest Classifier# Random
rf = RandomForestClassifier(n_estimators=100)
rf.fit(trainVector, train_data_label)
predictions = rf.predict(testVector)

print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(test_data_label, predictions)))
#modelEvaluation(predictions)
