from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from pyvi import ViTokenizer, ViPosTagger
from sklearn.base import TransformerMixin, BaseEstimator

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, *_):
        return self

    def transform(self, X, y=None, **fit_params):
        result = X.apply(lambda text: ViTokenizer.tokenize(text))
        return result
        
class SVMModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf-svm", SGDClassifier(loss='log', penalty='l2', alpha=1e-3, max_iter=5, random_state=None))
        ])

        return pipe_line
    
    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
