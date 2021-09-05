# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 13:25:14 2021

This module loads TFIDF vectorizer and trained model.
It predicts the topics given raw query data.

@author: 1353271 Mayur Kagathara
"""

import dill as pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

#%%
import os
import requests
from requests.api import request
curr_dir = os.getcwd()
voting_model_file = curr_dir+'/voting_model.sav'
title_vec_file = curr_dir+'/title_tfidf_vectorizer_6000f.sav'
abstract_vec_file = curr_dir+'/abstract_tfidf_vectorizer_6000f.sav'

if not os.path.isfile(voting_model_file):
    url = r'https://github.com/mayurkagathara/paper-classification/blob/main/output_files/voting_model.sav?raw=true'
    resp = requests.get(url)
    with open(voting_model_file, 'wb') as fopen:
        fopen.write(resp.content)

if not os.path.isfile(title_vec_file):
    url = r'https://github.com/mayurkagathara/paper-classification/blob/main/output_files/title_tfidf_vectorizer_6000f.sav?raw=true'
    resp = requests.get(url)
    with open(title_vec_file, 'wb') as fopen:
        fopen.write(resp.content)

if not os.path.isfile(abstract_vec_file):
    url = r'https://github.com/mayurkagathara/paper-classification/blob/main/output_files/abstract_tfidf_vectorizer_6000f.sav?raw=true'
    resp = requests.get(url)
    with open(abstract_vec_file, 'wb') as fopen:
        fopen.write(resp.content)


with open(voting_model_file, 'rb') as fread:
    clf_model_voting = pickle.load(fread)

#%%
# Global variables/objects
try:
    stop_words = stopwords.words('english')
except:
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

labels_ = ['Computer Science', 'Physics', 'Mathematics','Statistics', 'Quantitative Biology', 'Quantitative Finance']



title_vectorizer = pickle.load(open(title_vec_file,'rb'))
abstarct_vectorizer = pickle.load(open(abstract_vec_file,'rb'))


def clean_transform_title_abstract(dataframe, title_vectorizer, abstarct_vectorizer):
    '''
    Takes dataframe, title TFIDF vectorizer and abstract vectorizer as input.
    Returns tfidf vectorized title and abstract columns.
    This function only applies transformation.
    '''
    def text_clean(text):
        snowball_stemmer = SnowballStemmer("english")
        pattern = r'[^a-zA-Z0-9\s]'
        cleaned_sent = re.sub(pattern,'',text)
        word_tokens = cleaned_sent.split()
        word_tokens_stemmed = [snowball_stemmer.stem(w) for w in word_tokens if w not in stop_words]
        return ' '.join(word_tokens_stemmed)
    
    title_col = dataframe.TITLE
    cleaned_abstarct_col = dataframe.ABSTRACT.map(text_clean)
    
    title_col_tfidf = title_vectorizer.transform(title_col)
    abstract_col_tfidf = abstarct_vectorizer.transform(cleaned_abstarct_col)
    return title_col_tfidf, abstract_col_tfidf

def predict_topics(raw_data):
    title_col_vec_test, abstract_col_vec_test = clean_transform_title_abstract(raw_data, 
                                                                 title_vectorizer, 
                                                                 abstarct_vectorizer)
    # print(title_col_vec_test.shape)
    # print(abstract_col_vec_test.shape)
    
    title_features_test = pd.DataFrame(title_col_vec_test.todense(), columns=title_vectorizer.vocabulary_)
    abstract_features_test = pd.DataFrame(abstract_col_vec_test.todense(), columns=abstarct_vectorizer.vocabulary_)
    X_test = pd.concat([title_features_test, abstract_features_test], axis=1)
    
    y_pred = clf_model_voting.predict(X_test)
    predictions = pd.DataFrame(y_pred, columns=labels_)
    return predictions 

def predict_single(title, abstract):
    raw_data = pd.DataFrame([[title,abstract]], columns=['TITLE','ABSTRACT'])
    predictions_list = predict_topics_list(raw_data)
    return predictions_list.topics[0]

def convert_topics_list(predictions_dataframe):
    predictions_dataframe['topics'] = None
    label_array = np.array(labels_)
    for row in predictions_dataframe.T:
        label_out = np.array(predictions_dataframe.loc[row,labels_])
        predictions_dataframe.loc[row, 'topics'] = ', '.join(label_array[label_out.astype(bool)])
    return predictions_dataframe

def predict_topics_list(raw_data):
    predictions_dataframe = predict_topics(raw_data)
    return convert_topics_list(predictions_dataframe)
    
if __name__ == '__main__':
    pass