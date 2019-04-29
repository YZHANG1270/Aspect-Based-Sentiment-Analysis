# -*- coding: utf-8 -*-
__author__ = 'ZhangYi'

import os
import ast
import json
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from train.model.model import TextClassifier
from utils.utils import delimiter
from utils.data_process import nan_to_others,category_transpose,seg_words,load_aspect_list

class AspectClassifier(object):
    """
    Aspect(=EA) Classifier Train Part
    """
    def __init__(self):
        path_delimiter = delimiter()
        path_absa = os.path.abspath('..')

        # config
        task_tag = 'aspect_'
        model_name = task_tag + 'svc'

        # config path
        self.path_config = path_absa + path_delimiter + 'config.json'

        # model path
        self.model_path = path_absa + path_delimiter + 'model' + path_delimiter + '{}.mdl'.format(model_name)

        # data path
        self.path_data = path_absa + path_delimiter +'data'
        self.path_data_ch = path_absa + path_delimiter +'data' + path_delimiter + 'chinese' + path_delimiter
        self.path_train_df = self.path_data + path_delimiter + 'aspect' + path_delimiter + '{}_train.xlsx'.format(model_name)
        self.path_test_df = self.path_data + path_delimiter + 'aspect' + path_delimiter + '{}_test.xlsx'.format(model_name)

    def data_process(self):
        if os.path.isfile(self.path_train_df) \
                and os.path.isfile(self.path_test_df) \
                and os.path.isfile(self.path_config):

            train_df = pd.read_excel(self.path_train_df)
            test_df = pd.read_excel(self.path_test_df)
            self.category_list = load_aspect_list(self.path_config)

        else:
            # 1. load data
            train = pd.read_excel(self.path_data_ch+'Chinese_phones_training.xlsx')
            test = pd.read_excel(self.path_data_ch+'CH_PHNS_SB1_TEST.xlsx')

            # 2. mark NaN as 'OTHERS'
            _data = []
            for data in [train, test]:
                df = nan_to_others(data)
                _data.append(df)

            # 3. generate category list
            self.category_list = list(set(_data[0]['category']))  # len = 73

            # 4. save category list to config
            cate_dict = {'aspect_list':self.category_list}
            with open(self.path_config, "w") as f:
                f.write(json.dumps(cate_dict))
            f.close()

            # 5. generate df by category transpose
            all_data = []
            for d in _data:
                df = category_transpose(d, self.category_list)
                all_data.append(df)

            # 6. save data
            train_df, test_df = all_data[0], all_data[1]
            train_df.to_excel(self.path_train_df, index=False)
            test_df.to_excel(self.path_test_df, index=False)

        return train_df, test_df

    def train(self, train_df):
        content_train = seg_words(train_df['text'])
        vectorizer_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), min_df=5, norm='l2')
        vectorizer_tfidf.fit(content_train)

        # model train
        classifier_dict = dict()
        for column in self.category_list:
            print(column)
            label_train = train_df[column]
            text_classifier = TextClassifier(vectorizer=vectorizer_tfidf)
            text_classifier.fit(content_train, label_train)
            classifier_dict[column] = text_classifier

        # save model
        if os.path.isfile(self.model_path):
            pass
        else:
            joblib.dump(classifier_dict, self.model_path)

    def test(self, test_df):
        classifier = joblib.load(self.model_path)
        content_test = seg_words(test_df['text'])

        f1_score_dict = dict()
        for column in self.category_list:
            label_validate = test_df[column]
            text_classifier = classifier[column]
            f1_score = text_classifier.get_f1_score(content_test, label_validate)
            f1_score_dict[column] = f1_score

        f1_score = np.mean(list(f1_score_dict.values()))
        print('F1-SCORE-DICT: ', f1_score_dict)
        print('MEAN OF F1-SCORE-DICT: ', f1_score)

        return f1_score_dict


if __name__=="__main__":
    aspect = AspectClassifier()
    train_df, test_df = aspect.data_process()

    aspect.train(train_df)
    aspect.test(test_df)