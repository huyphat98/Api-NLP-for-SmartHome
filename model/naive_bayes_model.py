#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pyvi import ViTokenizer, ViPosTagger
from sklearn.base import TransformerMixin, BaseEstimator

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self

    def transform(self, X, y=None, **fit_params):
        result = X.apply(lambda text: ViTokenizer.tokenize(text))
        return result


class NaiveBayesModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),#sử dụng pyvi tiến hành word segmentation
            ("vect", CountVectorizer()),          #bag-of-words
            ("tfidf", TfidfTransformer()),        #tf-idf
            ("clf", MultinomialNB())              #model naive bayes
        ])

        return pipe_line