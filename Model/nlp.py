# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 00:42:06 2019

@author: clair
"""

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'




data = pd.read_csv("train/train.csv")

# extracting the number of examples of each class
EAP_len = data[data['author'] == 'EAP'].shape[0]
HPL_len = data[data['author'] == 'HPL'].shape[0]
MWS_len = data[data['author'] == 'MWS'].shape[0]
# bar plot of the 3 classes
plt.bar(10,EAP_len,3, label="Author EAP")
plt.bar(15,HPL_len,3, label="Author HPL")
plt.bar(20,MWS_len,3, label="Author MWS")
plt.legend()
plt.ylabel('Number of examples of authors')
plt.title('Propoertion of examples by authors')
plt.show()

"""

Feature engineering

"""
"""
 removing punctuation
"""
def remove_punctuation(text):
    import string
    translator = str.maketrans('','',string.punctuation)
    return text.translate(translator)

data['text'] = data['text'].apply(remove_punctuation)

"""
removing stopwords
"""
# extracting the stopwords from nltk library
from nltk.corpus import stopwords
sw = stopwords.words('english')
np.array(sw)

"""
removing stopwords
"""
def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)

data['text'] = data['text'].apply(stopwords)

"""

Top words before stemming

"""
"""
collect vocabulary count
"""
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
count_vectorizer.fit(data['text'])
dictionary = count_vectorizer.vocabulary_.items()  
vocab = []
count = []
for key, value in dictionary:
    vocab.append(key)
    count.append(value)
vocab_bef_stem = pd.Series(count, index=vocab)
vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)

"""
Stemming operations
Stemming operation bundles together words of same root. E.g. stem operation bundles "response" and "respond" into a common "respon"
"""
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)
data['text'] = data['text'].apply(stemming)

"""
Top words after stemming operation
"""
from sklearn.feature_extraction.text import TfidfVectorizer
tfid_vectorizer = TfidfVectorizer()
tfid_vectorizer.fit(data['text'])
dictionary = tfid_vectorizer.vocabulary_.items()
vocab = []
count = []
for key, value in dictionary:
    vocab.append(key)
    count.append(value)
vocab_after_stem  = pd.Series(count, index=vocab)
vocab_after_stem  = vocab_after_stem .sort_values(ascending=False)
top_vacab = vocab_after_stem.head(20)
top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (14980, 15020))

"""
histogram of text length of each writer
"""
def length(text):
    return len(text)
data['length'] = data['text'].apply(length)

"""
extracting data of each author
"""
EAP_data = data[data['author'] == 'EAP']
HPL_data = data[data['author'] == 'HPL']
MWS_data = data[data['author'] == 'MWS']

"""plotting histograms of text length of each author"""
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)
bins = 500
plt.hist(EAP_data['length'], alpha = 0.6, bins=bins, label='EAP')
plt.hist(HPL_data['length'], alpha = 0.8, bins=bins, label='HPL')
plt.hist(MWS_data['length'], alpha = 0.4, bins=bins, label='MWS')
plt.xlabel('length')
plt.ylabel('numbers')
plt.legend(loc='upper left')
plt.xlim(0,300)
plt.grid()
plt.show()

"""
Top words of each author and their count
"""
EAP_tfid_vectorizer = TfidfVectorizer("english")
EAP_tfid_vectorizer.fit(EAP_data['text'])
EAP_dictionary = EAP_tfid_vectorizer.vocabulary_.items()
vocab = []
count = []
for key, value in EAP_dictionary:
    vocab.append(key)
    count.append(value)
EAP_vocab = pd.Series(count, index=vocab)
EAP_vocab = EAP_vocab.sort_values(ascending=False)

HPL_tfid_vectorizer = TfidfVectorizer("english")
HPL_tfid_vectorizer.fit(HPL_data['text'])
HPL_dictionary = HPL_tfid_vectorizer.vocabulary_.items()
vocab = []
count = []
for key, value in HPL_dictionary:
    vocab.append(key)
    count.append(value)
HPL_vocab = pd.Series(count, index=vocab)
HPL_vocab = HPL_vocab.sort_values(ascending=False)

MWS_tfid_vectorizer = TfidfVectorizer("english")
MWS_tfid_vectorizer.fit(MWS_data['text'])
MWS_dictionary = MWS_tfid_vectorizer.vocabulary_.items()
vocab = []
count = []
for key, value in MWS_dictionary:
    vocab.append(key)
    count.append(value)
MWS_vocab = pd.Series(count, index=vocab)
MWS_vocab = MWS_vocab.sort_values(ascending=False)


"""
TF-IDF extraction
(normalized term frequency - inverse document frequency)
TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).
IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
word count
"""
tfid_matrix = tfid_vectorizer.transform(data['text'])
array = tfid_matrix.todense() #to numpy array
df = pd.DataFrame(array)


"""Part 3 training model"""
"""
Naive Bayes Classifier
- we have a medium size dataset
- it scales well
- it has been historically used in NLP tasks
Multinomial and Bernoulli NB classifier > Gaussian NB classifier
"""
df['output'] = data['author']
df['id'] = data['id']

features = df.columns.tolist()
output = 'output'
features.remove(output)
features.remove('id')


from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV

alpha_list1 = np.linspace(0.006, 0.1, 20)
alpha_list1 = np.around(alpha_list1, decimals=4)
"""Grid search"""
parameter_grid = [{"alpha":alpha_list1}]

classifier1 = MultinomialNB()
"""gridsearch object using 4 fold cross validation and neg_log_loss as scoring parameter"""
gridsearch1 = GridSearchCV(classifier1, parameter_grid, scoring='neg_log_loss', cv=4)
gridsearch1.fit(df[features], df[output])

#results1 = pd.DataFrame()
## collect alpha list
#results1['alpha'] = gridsearch1.cv_results_['param_alpha'].data
## collect test scores
#results1['neglogloss'] = gridsearch1.cv_results_['mean_test_score'].data
#
#matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
#plt.plot(results1['alpha'], -results1['neglogloss'])
#plt.xlabel('alpha')
#plt.ylabel('logloss')
#plt.grid()
#plt.show()














