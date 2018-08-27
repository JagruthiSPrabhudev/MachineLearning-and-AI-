import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from gensim.models import word2vec

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# read the data from csv
reviews = pd.read_csv('reviews.csv', error_bad_lines=False, sep='|')
print("Totalreviews :", len(reviews))
# get the length of the reviews
reviews['text length'] = reviews['text'].apply(len)

# convert labels positive to 1 and negative to 0
reviews['label'] = np.where(reviews['label'] == 'positive', 1, 0)

X = reviews['label']
Y = reviews['text']

# change index of review to start from 1
reviews.index = reviews.index + 1

# separate train data and test data
test_data = reviews[reviews.index % 5 == 0]
train_data = reviews[reviews.index % 5 != 0]
print("test data", len(test_data))
print("train data", len(train_data))
train_data_reviews = train_data['text']
train_data_label = train_data['label']
test_data_reviews = test_data['text']
test_data_label = test_data['label']

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


X_train_cleaned = []
X_test_cleaned = []

for d in train_data_reviews:
    X_train_cleaned.append(cleanText(d))
print('Show a cleaned review in the training set : \n', X_train_cleaned[10])

for d in test_data_reviews:
    X_test_cleaned.append(cleanText(d))

tfidf = TfidfVectorizer(min_df=5) #minimum document frequency of 5
X_train_tfidf = tfidf.fit_transform(train_data_reviews)

estimators = [("tfidf", TfidfVectorizer()), ("lr", LogisticRegression())]
model = Pipeline(estimators)
# Grid search
params = {"lr__C":[0.1, 1, 10], #regularization param of logistic regression
          "tfidf__min_df": [1, 3], #min count of words
          "tfidf__max_features": [1000, None], #max features
          "tfidf__ngram_range": [(1,1), (1,2)], #1-grams or 2-grams
          "tfidf__stop_words": [None, "english"]} #use stopwords or don't

grid = GridSearchCV(estimator=model, param_grid=params, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_tfidf, train_data_label)
print("The best paramenter set is : \n", grid.best_params_)


# Evaluate on the validaton set
predictions = grid.predict(X_test_cleaned)

print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(test_data_label, predictions)))
