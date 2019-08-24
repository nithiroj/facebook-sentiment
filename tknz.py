# -*- coding: utf-8 -*

import keras
import html
from html.parser import HTMLParser
import nltk
import pandas as pd
import numpy as np
import re
import glob
import codecs
from pythainlp.tokenize import dict_word_tokenize, word_tokenize
import deepcut

from sklearn.externals import joblib

DICT = './my_dict.txt'
STOPWORDS = './my_stopwords.txt'

'''def dummy_fun(doc):  # to apply scikit-learn CountVectorizer, TfidfVectorizer on tokenized text
    return doc'''

# Load stopwords
def load_my_stopwords(file=STOPWORDS):
    with codecs.open(file, 'r', encoding='utf8', errors='replace') as f:
        my_stopwords = f.read().splitlines()
        my_stopwords[0] = my_stopwords[0].replace(u'\ufeff', '')  # Remove U+FEFF is the Byte Order Mark character in first word.
        return my_stopwords

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

# Stripe HTML
def strip_html(text):
    html_stripper = HTMLStripper()
    html_stripper.feed(text)
    return html_stripper.get_data()

# Remove 'Attached Description :' and 'Attached Story :'
def strip_text(text):
    text = re.sub(r'Attached Description :',  '', text)
    text = re.sub(r'Attached Story :',  '', text)
    text = re.sub(r'Timeline Photos',  '', text)
    text = re.sub(r'[^ก-๙]',  '', text)  # () A-Za-z0-9.,!?@\'\`\"\_\n
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', '', text)
    return text

def deepcut_segment(text, data=""):
	if data=="":
		return deepcut.tokenize(text)
	else:
		word_list = list(set(data))
		return deepcut.tokenize(text, word_list)

def deepcut_tokenize(text,file='',data=[''],data_type="file"):
    if data_type=='file':
        with codecs.open(file, 'r',encoding='utf8') as f:
            lines = f.read().splitlines()
            f.close()
    elif data_type=='list':
        lines = data
    
    return deepcut_segment(text,data=lines)
    
# Tokenize function
def tokenize(text, engine='deepcut'):
    if engine == 'deepcut':
        # tokens = dict_word_tokenize(text, file=DICT, engine=engine)
        tokens = deepcut_tokenize(text, file=DICT)
    else:
        tokens = word_tokenize(text)
    tokens = list(filter(str.strip, tokens))
    return tokens

def normalize_tokens(text, stopwords=False):
    text = html.unescape(text)
    text = strip_html(text)
    text = strip_text(text)
    tokens = tokenize(text)
    if len(tokens) > 0:
        if '...' in tokens[-1]:
            tokens.pop()
    tokens = [token for token in tokens if len(token) > 1]
    if stopwords:
        stopword_list = load_my_stopwords()
        tokens = [token for token in tokens if token not in stopword_list]
    return tokens
