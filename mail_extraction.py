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
import difflib
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
    frame = pandas.read_csv('data/result_frame.csv', encoding='cp1252')

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

    ## Additional part from kaggle 0.33

    # stops = set(stopwords.words("english"))
    # def get_unigrams(que):
    #     return [word for word in nltk.word_tokenize(que.lower()) if word not in stops]
    #
    # def get_common_unigrams(row):
    #     return len(set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])))
    #
    # def get_common_unigram_ratio(row):
    #     return float(row["zunigrams_common_count"]) / max(
    #         len(set(row["unigrams_ques1"]).union(set(row["unigrams_ques2"]))), 1)
    #
    # def get_bigrams(que):
    #     return [i for i in nltk.ngrams(que, 2)]
    #
    # def get_common_bigrams(row):
    #     return len(set(row["bigrams_ques1"]).intersection(set(row["bigrams_ques2"])))
    #
    # def get_common_bigram_ratio(row):
    #     return float(row["zbigrams_common_count"]) / max(
    #         len(set(row["bigrams_ques1"]).union(set(row["bigrams_ques2"]))), 1)
    #
    # frame['question1_nouns'] = frame.question1.map(
    #     lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    # frame['question2_nouns'] = frame.question2.map(
    #     lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    #
    # frame['z_len1'] = frame.question1.map(lambda x: len(str(x)))
    # frame['z_len2'] = frame.question2.map(lambda x: len(str(x)))
    # frame['z_word_len1'] = frame.question1.map(lambda x: len(str(x).split()))
    # frame['z_word_len2'] = frame.question2.map(lambda x: len(str(x).split()))
    # frame['z_noun_match'] = frame.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)
    #
    # frame["unigrams_ques1"] = frame['question1'].apply(lambda x: get_unigrams(str(x)))
    # frame["unigrams_ques2"] = frame['question2'].apply(lambda x: get_unigrams(str(x)))
    # frame["zunigrams_common_count"] = frame.apply(lambda r: get_common_unigrams(r), axis=1)
    # frame["zunigrams_common_ratio"] = frame.apply(lambda r: get_common_unigram_ratio(r), axis=1)
    # frame["bigrams_ques1"] = frame["unigrams_ques1"].apply(lambda x: get_bigrams(x))
    # frame["bigrams_ques2"] = frame["unigrams_ques2"].apply(lambda x: get_bigrams(x))
    # frame["zbigrams_common_count"] = frame.apply(lambda r: get_common_bigrams(r), axis=1)
    # frame["zbigrams_common_ratio"] = frame.apply(lambda r: get_common_bigram_ratio(r), axis=1)
    #
    #
    #
    #
    #
    #
    # def diff_ratios(s):
    #     seq = difflib.SequenceMatcher()
    #     seq.set_seqs(str(s['question1']).lower(), str(s['question2']).lower())
    #     return seq.ratio()
    #
    # frame['diff'] = frame[['question1', 'question2']].apply(diff_ratios, axis=1)

    #E.saver(frame, 'train_res_uni_zen.csv')

    print('                  Rake:', roc_auc_score(frame['is_duplicate'], frame['rake'].fillna(0)))
    print('                 Tfidf:', roc_auc_score(frame['is_duplicate'], frame['tfidf'].fillna(0)))
    print('                  Dist:', roc_auc_score(frame['is_duplicate'], frame['dist'].fillna(0)))
    print('               Rake_my:', roc_auc_score(frame['is_duplicate'], frame['rake_my'].fillna(0)))
    print('                  Comp:', roc_auc_score(frame['is_duplicate'], frame['comp'].fillna(0)))
    print('                  Diff:', roc_auc_score(frame['is_duplicate'], frame['diff'].fillna(0)))


    print('zunigrams_common_ratio:', roc_auc_score(frame['is_duplicate'], frame['zunigrams_common_ratio'].fillna(0)))
    print(' zbigrams_common_ratio:', roc_auc_score(frame['is_duplicate'], frame['zbigrams_common_ratio'].fillna(0)))
    print('zunigrams_common_count:', roc_auc_score(frame['is_duplicate'], frame['zunigrams_common_count'].fillna(0)))
    print(' zbigrams_common_count:', roc_auc_score(frame['is_duplicate'], frame['zbigrams_common_count'].fillna(0)))

    # print('z_len1:', roc_auc_score(frame['is_duplicate'], frame['z_len1'].fillna(0)))
    # print('z_len2:', roc_auc_score(frame['is_duplicate'], frame['z_len2'].fillna(0)))
    # print('z_word_len1:', roc_auc_score(frame['is_duplicate'], frame['z_word_len1'].fillna(0)))
    # print('z_word_len2:', roc_auc_score(frame['is_duplicate'], frame['z_word_len2'].fillna(0)))
    # print('z_noun_match:', roc_auc_score(frame['is_duplicate'], frame['z_noun_match'].fillna(0)))
    # print('unigrams_ques1:', roc_auc_score(frame['is_duplicate'], frame['unigrams_ques1'].fillna(0)))
    # print('unigrams_ques2:', roc_auc_score(frame['is_duplicate'], frame['unigrams_ques2'].fillna(0)))
    # print('bigrams_ques1:', roc_auc_score(frame['is_duplicate'], frame['bigrams_ques1'].fillna(0)))
    # print('bigrams_ques2:', roc_auc_score(frame['is_duplicate'], frame['bigrams_ques2'].fillna(0)))
    # #regr = CustomEnsembleRegressor([en.GradientBoostingClassifier()])

    x_train = pandas.DataFrame()
    x_test = pandas.DataFrame()
    x_train['rake'] = frame['rake']
    x_train['tfidf'] = frame['tfidf']
    x_train['dist'] = frame['dist']
    x_train['comp'] = frame['comp']
    x_train['diff'] = frame['diff']

    x_train['zunigrams_common_ratio'] = frame['zunigrams_common_ratio']
    x_train['zbigrams_common_ratio'] = frame['zbigrams_common_ratio']
    x_train['zunigrams_common_count'] = frame['zunigrams_common_count']
    x_train['zbigrams_common_count'] = frame['zbigrams_common_count']

    # x_train['z_len1'] = frame['z_len1']
    # x_train['z_len2'] = frame['z_len2']
    # x_train['z_word_len1'] = frame['z_word_len1']
    # x_train['z_word_len2'] = frame['z_word_len2']
    # x_train['unigrams_ques1'] = frame['unigrams_ques1']
    # x_train['unigrams_ques2'] = frame['unigrams_ques2']
    # x_train['bigrams_ques1'] = frame['bigrams_ques1']
    # x_train['bigrams_ques2'] = frame['bigrams_ques2']
    # x_train['z_noun_match'] = frame['z_noun_match']

    # x_test['word_match'] = frame_test.apply(word_match_share, axis=1, raw=True)
    # x_test['tfidf_word_match'] = frame_test.apply(tfidf_word_match_share, axis=1, raw=True)

    y_train = frame['is_duplicate'].values

    pos_train = x_train[y_train == 1]
    neg_train = x_train[y_train == 0]

    # Now we oversample the negative class
    # There is likely a much more elegant way to do this...
    # p = 0.2
    # scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    # while scale > 1:
    #     neg_train = pandas.concat([neg_train, neg_train])
    #     scale -= 1
    # neg_train = pandas.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    # print(len(pos_train) / (len(pos_train) + len(neg_train)))

    x_train = pandas.concat([pos_train, neg_train])
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
    del pos_train, neg_train

    #x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    x_train['is_duplicate'] = y_train
    print(Learning(x_train[['comp',
                            'rake',
                            'dist',
                            'diff',
                            'tfidf',
                            'zbigrams_common_count',
                            'zunigrams_common_count',
                            'zbigrams_common_ratio',
                            'zunigrams_common_ratio',
                            'is_duplicate']], y_col='is_duplicate').trees(m_params={
        'verbose': True,
        'criterion': 'mse',
        'n_estimators': 2500,
        'learning_rate': 0.07
    },
        models=en.GradientBoostingClassifier))

if __name__ == '__main__':
    go()
