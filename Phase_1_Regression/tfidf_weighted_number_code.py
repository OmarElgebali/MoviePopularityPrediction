import pandas as pd
import numpy as np
import nltk
import json
import string
from datetime import datetime

import scipy
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import sklearn
import csv
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse import csr_matrix

movies_dataset = pd.read_csv('movies-regression-dataset.csv')

column_name = 'title'

def w_avg(arr):
    weight = 0  # weight
    product = 0  # position*weight
    indices = arr.indices
    data = arr.data
    product = np.dot(data, indices)
    weight = data.sum()
    if weight == 0:
        return 0
    return product / weight + 1  # weighted average

v = []
for index, r in enumerate(movies_dataset[column_name]):
    t = nltk.word_tokenize(r)
    lemmatized_text = ' '.join(t)
    v.append(lemmatized_text)
movies_dataset[column_name] = v

tfidf = TfidfVectorizer()
tfidf.fit(movies_dataset[column_name])
tfidf_data = tfidf.transform(movies_dataset[column_name])

for row in [85, 403]:
    print(tfidf_data.getrow(row))
    print(tfidf.inverse_transform(tfidf_data.getrow(row)))
    print(w_avg(tfidf_data.getrow(row)))
    print('='*100)


