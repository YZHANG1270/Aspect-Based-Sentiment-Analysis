#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'ZhangYi'

import ast
import jieba
import itertools
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# mark NaN as 'OTHERS'
def nan_to_others(df):
    new_cate = []
    new_polarity = []

    # dataframe必须含有列：['text', 'category', 'polarity']
    for idx, i in enumerate(df['polarity']):
        if i in ['negative', 'positive', 'neutral', 'conflict']:
            new_cate.append(df['category'][idx])
            new_polarity.append(i)
        else:
            new_cate.append('OTHERS')
            new_polarity.append('OTHERS')
    _df = pd.DataFrame(np.array([df['text'], new_cate, new_polarity]).T, columns=['text', 'category', 'polarity'])
    return _df

# tokenize
def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content)
        contents_segs.append(" ".join(segs))
    return contents_segs

# get text vector
def gen_text_vec(tokenizer, cut_corpus_list, maxlen):
    text_vec = tokenizer.texts_to_sequences(cut_corpus_list)
    t_vec = pad_sequences(text_vec, maxlen=maxlen)
    return t_vec

# category transpose
def category_transpose(df, category_list):
    for i in category_list:
        l_ist = []
        # dataframe必须含有列：['category']
        for cate in df['category']:
            if cate == i:
                l_ist.append(1)
            else:
                l_ist.append(0)
        df[i] = l_ist
    return df

# load config: aspect_list
def load_aspect_list(path_config):
    # only one param in config: aspect_list
    a = 0
    with open(path_config, "r", encoding='utf-8') as f:
        for i in f:
            category_list = ast.literal_eval(i)['aspect_list']
            a = a + 1
            if a == 1:
                break
    f.close()
    return category_list

# merge excel
def merge_excel(path_data_dir):
    cmt_l = []
    scr_l = []

    # 被merge的df都必须有['comment_content', 'label']
    data_source = ['/2019-04-12_lock_comment_jd_spider_baidu_sentiment.xlsx', \
                   '/20190329_train_lock_comments_document_level_with_label.xls', \
                   '/all_comments_document_level_without_lock_comments.xls', \
                   '/bad_comments_in_forum_mi.com_youpin.xls']
    for i in data_source:
        path_data = path_data_dir + i
        _data = pd.read_excel(path_data)

        cmt_l.append(_data['comment_content'])
        scr_l.append(_data['label'])

    comment = list(itertools.chain.from_iterable(cmt_l))
    score = list(itertools.chain.from_iterable(scr_l))

    data = pd.DataFrame(np.array([comment, score]).T, columns=['comment_content', 'label'])
    return data

# remove row by column with empty value
def remove_empty_row(df, column_name):
    row_to_delete = []
    for idx, i in enumerate(df[column_name]):
        if not bool(i):
            row_to_delete.append(idx)
    df = df.drop(df.index[row_to_delete])
    return df.reset_index(drop=True)