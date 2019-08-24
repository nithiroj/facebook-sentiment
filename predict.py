# -*- coding: utf-8 -*

from tknz import *
from dummy import dummy_fun

# pipe = joblib.load('ros_lrg_ns_bow_pipe.joblib')

def predict(text, model='ros_lrg', stopwords=False):
    preds = []
    token = normalize_tokens(text, stopwords)
    print(f'Text: {text}')
    print(f'Tokens: {token}')
    subfix = '' if stopwords else '_ns'
    pipe = joblib.load(f'{model}{subfix}_bow_pipe.joblib')
    pred = pipe.predict([token])

    return pred, token

if __name__ == "__main__":
    pred, token = predict('พี่ตูนเยี่ยมสุดๆ')
    print(pred)