# -*- coding: utf-8 -*-

__author__ = 'ZhangYi'

import sys

def delimiter():
    path_delimiter = '/'
    if 'win' in sys.platform:
        path_delimiter = '\\'
    return path_delimiter