#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'ZhangYi'

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout,Bidirectional, GlobalMaxPool1D


class BiLSTM():
    def __init__(self, max_features, embed_size):
        model = Sequential()
        model.add(Embedding(max_features, embed_size))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(GlobalMaxPool1D())
        model.add(Dense(20, activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.classifier = model

    def fit(self, x, y, batch_size, epochs, validation_split):
        self.classifier.fit(x,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    def predict(self, x):
        return self.classifier.predict(x)

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_pred, y_true)
        f1 = f1_score(y_pred, y_true)
        cfs_matrix = confusion_matrix(y_pred, y_true)
        print('Accuracy Score:', acc)
        print('F1-score: {0}'.format(f1))
        print('Confusion matrix:\n', cfs_matrix)

        return acc, f1, cfs_matrix

    def _make_predict_function(self):
        self.classifier._make_predict_function()