# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 00:42:06 2019

@author: clair
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from nltk.stem.snowball import SnowballStemmer
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
count_vec
