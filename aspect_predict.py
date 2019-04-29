# -*- coding: utf-8 -*-

__author__ = 'ZhangYi'

import os
from sklearn.externals import joblib

from utils.utils import delimiter
from utils.data_process import seg_words,load_aspect_list

class AspectPredict(object):
    def __init__(self):
        path_delimiter = delimiter()
        path_absa = os.path.abspath('.')

        # config
        model_name = 'aspect_svc' # todo: add to config
        path_config = path_absa + path_delimiter + 'config.json'

        # load model
        path_model = path_absa + path_delimiter + 'model' + path_delimiter + '{}.mdl'.format(model_name)
        self.model = joblib.load(path_model)

        # load aspect list
        self.aspect_list = load_aspect_list(path_config)


    def predict(self, text):
        # 1. generate result
        result = dict()
        result['text'] = text
        result['aspectCategory'] = []

        # 2. seg words
        content_test = seg_words([text])

        # 3. predict
        all_result = dict()
        for column in self.aspect_list:
            all_result[column] = self.model[column].predict(content_test)[0]
            if all_result[column]>0.5:
                result['aspectCategory'].append(column)
        result['all_result'] = all_result

        print('PREDICT RESULT:',result)
        print('PREDICT ASPECT:', result['aspectCategory'])
        return result

if __name__=="__main__":
    aspect = AspectPredict()
    aspect.predict('这块屏幕不错')