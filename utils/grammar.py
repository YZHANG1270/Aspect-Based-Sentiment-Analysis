#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'ZhangYi'

import re

def chinese_only(txt_list):
    cmt_zh = []
    for cmt in txt_list:
        line = cmt.strip()
        p2 = re.compile(u'[^\u4e00-\u9fa5]')
        zh = " ".join(p2.split(line)).strip()
        cmt_zh.append(",".join(zh.split()))
    return cmt_zh