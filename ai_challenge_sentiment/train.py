# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
os.chdir("C:/Users/LUMI/Desktop/sentiment")

import pandas as pd
import jieba
from model import TextClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.externals import joblib


def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content)
        contents_segs.append(" ".join(segs))

    return contents_segs

# load train data
train_data_df = pd.read_csv('data/train/train.csv')
validate_data_df = pd.read_csv('data/validation/validation.csv')

content_train = train_data_df.iloc[:, 1]
content_train = seg_words(content_train)

columns = train_data_df.columns.values.tolist()

vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2')
vectorizer_tfidf.fit(content_train)

# model train
classifier_dict = dict()
for column in columns[2:]:
    label_train = train_data_df[column]
    text_classifier = TextClassifier(vectorizer=vectorizer_tfidf)
    text_classifier.fit(content_train, label_train)
    classifier_dict[column] = text_classifier


# validate model
content_validate = validate_data_df.iloc[:, 1]

content_validate = seg_words(content_validate)


f1_score_dict = dict()
for column in columns[2:]:
    label_validate = validate_data_df[column]
    text_classifier = classifier_dict[column]
    f1_score = text_classifier.get_f1_score(content_validate, label_validate)
    f1_score_dict[column] = f1_score

f1_score = np.mean(list(f1_score_dict.values()))
str_score = "\n"
for column in columns[2:]:
    str_score = str_score + column + ":" + str(f1_score_dict[column]) + "\n"

# save model
joblib.dump(classifier_dict, model_save_path + model_name)

