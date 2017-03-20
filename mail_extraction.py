from data_extraction import Extractor, Learning
import pandas
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.metrics import log_loss, make_scorer
import numpy as np
import sklearn.ensemble as en
import sklearn.linear_model as ln
import sklearn.neighbors as ne
import sklearn.svm as S
import sklearn.tree as tr
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


from sklearn.base import ClassifierMixin
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, classificators=None):
        self.classificators = classificators

    def fit(self, X, y):
        for classificator in self.classificators:
            classificator.fit(X, y)

    def predict_proba(self, X):
        self.predictions_ = list()
        for classificator in self.classificators:
            # for classificator
            self.predictions_.append(classificator.predict_proba(X)[::,1].ravel())
            # for regressor
            #self.predictions_.append(classificator.predict(X).ravel())

        return np.mean(self.predictions_, axis=0)


class RandomForestClassifier_compability(ln.LogisticRegression):
    def predict(self, X):
        return self.predict_proba(X)[:, 1][:, np.newaxis]

def go():

    E = Extractor(work_dir='./',
                  file_train='data/train.csv')
    frame = E.df_creation()

    # from ngram import NGram
    # import pylev

    # def my_compare(s):
    #     try:
    #         return NGram.compare(s['question1'], s['question2'])
    #     except TypeError:
    #         return np.nan
    #
    # def my_dist(s):
    #     try:
    #         return 1/pylev.levenshtein(s['question1'], s['question2'])
    #     except TypeError:
    #         return np.nan
    #
    #
    # #frame = frame.head(100)
    #
    # #frame['comp'] = frame[['question1', 'question2']].apply(my_compare, axis=1)
    # #frame['dist'] = frame[['question1', 'question2']].apply(my_dist, axis=1)
    #
    # import RAKE
    # Rake = RAKE.Rake('stop.txt')
    #
    # def word_match_share(row):
    #     q1words = {}
    #     q2words = {}
    #     for word in str(row['question1']).lower().split():
    #         if word not in open('stop.txt', 'r').readlines():
    #             q1words[word] = 1
    #     for word in str(row['question2']).lower().split():
    #         if word not in open('stop.txt', 'r').readlines():
    #             q2words[word] = 1
    #     if len(q1words) == 0 or len(q2words) == 0:
    #         return 0
    #     shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    #     shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    #     R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    #     return R
    #
    # #frame['rake'] = frame.apply(word_match_share, axis=1, raw=True)
    #
    # def rake_via_rake(row):
    #     rk1 = Rake.run(str(row['question1']).lower())
    #     rk2 = Rake.run(str(row['question2']).lower())
    #     rk1_ln = len(rk1)
    #     rk2_ln = len(rk2)
    #     rk1 = map(lambda x: x[1], rk1)
    #     rk2 = map(lambda x: x[1], rk2)
    #     try:
    #         return (rk1_ln+rk2_ln)/(sum(rk1) + sum(rk2))
    #     except ZeroDivisionError:
    #         return 1
    #
    # #frame['rake_my'] = frame.apply(rake_via_rake, axis=1, raw=True)
    #
    # from collections import Counter
    #
    # def get_weight(count, eps=10000, min_count=2):
    #     if count < min_count:
    #         return 0
    #     else:
    #         return 1 / (count + eps)
    #
    # eps = 5000
    # train_qs = pandas.Series(frame['question1'].tolist() + frame['question2'].tolist()).astype(str)
    # words = (" ".join(train_qs)).lower().split()
    # counts = Counter(words)
    # weights = {word: get_weight(count) for word, count in counts.items()}
    #
    # def tfidf_word_match_share(row):
    #     q1words = {}
    #     q2words = {}
    #     for word in str(row['question1']).lower().split():
    #         if word not in open('stop.txt', 'r').readlines():
    #             q1words[word] = 1
    #     for word in str(row['question2']).lower().split():
    #         if word not in open('stop.txt', 'r').readlines():
    #             q2words[word] = 1
    #     if len(q1words) == 0 or len(q2words) == 0:
    #         # The computer-generated chaff includes a few questions that are nothing but stopwords
    #         return 0
    #
    #     shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
    #                                                                                     q2words.keys() if w in q1words]
    #     total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    #
    #     R = np.sum(shared_weights) / np.sum(total_weights)
    #     return R
    #
    # frame['tfidf'] = frame.apply(tfidf_word_match_share, axis=1, raw=True)

    #frame[['comp', 'dist', 'rake', 'rake_my', 'tfidf']] = np.log1p(frame[['comp', 'dist', 'rake', 'rake_my', 'tfidf']])

    #frame[['id', 'is_duplicate', 'comp', 'dist', 'rake', 'rake_my', 'tfidf']].plot.bar(x='id')
    #plt.show()


    print('   Rake:', roc_auc_score(frame['is_duplicate'], frame['rake'].fillna(0)))
    print('  Tfidf:', roc_auc_score(frame['is_duplicate'], frame['tfidf'].fillna(0)))
    print('   Dist:', roc_auc_score(frame['is_duplicate'], frame['dist'].fillna(0)))
    print('Rake_my:', roc_auc_score(frame['is_duplicate'], frame['rake_my'].fillna(0)))
    print('   Comp:', roc_auc_score(frame['is_duplicate'], frame['comp'].fillna(0)))
    regr = CustomEnsembleRegressor([en.GradientBoostingClassifier()])
    # print(Learning(frame.drop(['id', 'qid1', 'qid2', 'question1', 'question2', 'rake_my'], axis=1), y_col='is_duplicate').
    #       trees(m_params={#'verbose': True,
    #                       'classificators': [en.GradientBoostingClassifier(n_estimators=10)]
    #                       #'criterion': 'mse'
    #                      }, models=CustomEnsembleRegressor))

if __name__ == '__main__':
    go()
