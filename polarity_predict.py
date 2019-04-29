#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'ZhangYi'

import os
from sklearn.externals import joblib

from utils.utils import delimiter
from utils.grammar import chinese_only
from utils.data_process import seg_words, gen_text_vec

class PolarityClassifier(object):
    """
    text classification
    """
    def __init__(self):
        # config
        model_name = 'polarity_doc' # doc-based

        # path
        path_delimiter = delimiter()
        if 'absa' in os.path.abspath('.').split(path_delimiter):
            path_absa = os.path.abspath('.')
        else:
            # 被调用路径=path_comment
            path_absa = os.path.abspath('.') + path_delimiter + 'train' \
                        + path_delimiter + 'sentiment' + path_delimiter + 'absa'

        # model path
        path_model_dir = path_absa + path_delimiter + 'model'

        # load tokenizer
        path_tokenizer = path_model_dir + path_delimiter + '{}.tk'.format(model_name)
        self.tokenizer = joblib.load(path_tokenizer)

        # load model
        path_model = path_model_dir + path_delimiter + '{}.mdl'.format(model_name)
        self.model = joblib.load(path_model)
        self.model._make_predict_function()

    def predict(self, comment):

        # 1. chinese only
        cmt = chinese_only([comment])

        # 2. jieba token
        cmt = seg_words(cmt)[0]

        # 3. gen word vector
        _cmt = gen_text_vec(self.tokenizer, cmt, maxlen = 200)

        # token observation
        # split_tokens = []
        # for token in str(_cmt).split(" "):
        #     if token.isdigit():
        #         split_tokens.append(token)
        # print("len(split_tokens):{}".format(len(split_tokens)))

        # 4. predict
        neg_prob = self.model.predict(_cmt)[0][0]
        # neg_prob = (neg_prob > 0.5)

        # 5. json result output
        result = {'items':[{'negative_prob': 0,'sentiment': 0}], 'log_id': '', 'text': ''}
        result['items'][0]['negative_prob'] = neg_prob
        result['items'][0]['sentiment'] = int(round(neg_prob))  # 1表示差评；0表示好评
        result['text'] = comment
        print("SENTIMENT RESULT: ",result)
        return result


if __name__=="__main__":
    t = PolarityClassifier()
    t.predict('这块电池好看')
