#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'ZhangYi'

import os
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


from train.model.bilstm import BiLSTM
from utils.utils import delimiter
from utils.grammar import chinese_only
from utils.data_process import merge_excel,seg_words,remove_empty_row,gen_text_vec

class PolarityClassifier(object):
    """
    train sentiment model and generate model file
    """
    def __init__(self):
        path_delimiter = delimiter()
        path_absa = os.path.abspath('..')

        # config
        self.maxlen = 200          # doc word length
        task_tag = 'polarity_'
        model_name = task_tag + 'docu'

        # model path
        path_model = path_absa + path_delimiter + 'model'
        self.model_path = path_model + path_delimiter + '{}.mdl'.format(model_name)
        self.path_tokenizer = path_model + path_delimiter + '{}.tk'.format(model_name)

        # data path
        path_data_doc_level = path_delimiter.join(path_absa.split(path_delimiter)[:-2]) + path_delimiter + "data" \
                              + path_delimiter + 'sentiment' + path_delimiter + 'document_level'
        self.path_train_data = path_data_doc_level + path_delimiter + 'train_data'

        self.path_data = path_absa + path_delimiter + 'data'
        self.path_corpus = self.path_data + path_delimiter + 'polarity' + path_delimiter + '{}.xlsx'.format(model_name)

        # generate tokenizer
        self.data = self.data_process()
        self.tokenizer = self.gen_tokenizer(self.data['cmt_split'])

    def data_process(self):
        if os.path.isfile(self.path_corpus):
            data = pd.read_excel(self.path_corpus)

        else:
            # 1. merge data
            data = merge_excel(self.path_train_data)

            # 2. Chinese character only
            data['cmt_zh'] = chinese_only(data['comment_content'])

            # 3. jieba token for dictionary
            data['cmt_split'] = seg_words(data['cmt_zh'])

            # 4. remove empty comment
            data = remove_empty_row(data, 'cmt_split')

            # 5. save data
            data.to_excel(self.path_corpus)
        return data

    def gen_tokenizer(self, cut_corpus_list):
        if os.path.isfile(self.path_tokenizer):
            tokenizer = joblib.load(self.path_tokenizer)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(cut_corpus_list.astype(str))
            joblib.dump(tokenizer, self.path_tokenizer)
        return tokenizer

    def gen_train_test(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        embed_size = 256
        max_features = 66000  # dictionary size
        classifier = BiLSTM(max_features, embed_size)

        epochs = 2
        batch_size = 100
        X_tr = gen_text_vec(self.tokenizer, X_train, self.maxlen)
        classifier.fit(X_tr, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

        # save model
        if os.path.isfile(self.model_path):
            print('model exist already')
        else:
            joblib.dump(classifier, self.model_path)

    def test(self, X_test, y_test):
        X_te = gen_text_vec(self.tokenizer, X_test, self.maxlen)

        # load model
        model = joblib.load(self.model_path)
        pred_prob = model.predict(X_te)

        # pred = (pred_prob > 0.65)
        pred = [int(round(i[0])) for i in pred_prob]
        y_test = [int(i) for i in y_test]

        # evaluate
        eval = model.evaluate(y_test, pred)
        return eval


    def batch_predict(self, batch_cmt_df):

        # 1. chinese only
        batch_cmt_df['cmt_zh'] = chinese_only(batch_cmt_df['comment_content'])

        # 2. token cut
        batch_cmt_df['cmt_split'] = seg_words(batch_cmt_df['cmt_zh'])

        # 暂时没有remove empty环节

        # 3. predict
        self.test(batch_cmt_df['cmt_split'],batch_cmt_df['label'])

        # # save result
        # result = pd.DataFrame(np.array([self.X_test,self.y_test,pred]).T,columns=['comment_zh','GroundTruth','bilstm'])
        # result.to_excel('data/sentiment/result_.xlsx')


if __name__=="__main__":
    pc = PolarityClassifier()
    data = pc.data
    X_train, X_test, y_train, y_test = pc.gen_train_test(data['cmt_split'], data['label'])
    pc.train(X_train, y_train)
    pc.test(X_test, y_test)