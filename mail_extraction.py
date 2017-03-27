from data_extraction import Extractor, Learning
import pandas
import numpy as np
import sklearn.ensemble as en
import sklearn.linear_model as ln
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.style.use('ggplot')

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
    frame = pandas.read_csv('all_and_my_mutalist.csv', encoding='cp1252')

    sp = ['comp', 'rake', 'dist',
          'diff', 'tfidf', 'cosine',
          'muta', 'mnk', 'zbigrams_common_count',
          'zunigrams_common_count', 'zbigrams_common_ratio', 'zunigrams_common_ratio']

    ls = []
    for i in sp:
        for j in sp:
            if i != j:
                exec('frame["{0}/{1}"] = frame["{0}"]/frame["{1}"]'.format(i, j))
                exec('frame["{0}*{1}"] = frame["{0}"]*frame["{1}"]'.format(i, j))
                ls.append('{0}/{1}'.format(i, j))
                ls.append('{0}*{1}'.format(i, j))

        sp.remove(i)
    print(frame.head(20))

    frame = frame.replace([np.inf, -np.inf], np.nan)

    # def nth_root(value, n_root):
    #     root_value = 1 / float(n_root)
    #     return round(Decimal(value) ** Decimal(root_value), 3)
    #
    # def text_to_vector(text):
    #     try:
    #         words = text.split(' ')
    #     except AttributeError:
    #         return Counter(''.split(' '))
    #     return Counter(words)
    #
    # def my_mutalist(s):
    #     x = text_to_vector(s['question1'])
    #     y = text_to_vector(s['question2'])
    #     vect = DictVectorizer(sparse=False).fit_transform([x,y])
    #     sm = sum(abs(a - b) for a, b in vect.T)
    #     if sm:
    #         return 1/sm
    #     else:
    #         return 1
    #
    # def my_mutamnk(s):
    #     x = text_to_vector(s['question1'])
    #     y = text_to_vector(s['question2'])
    #     vect = DictVectorizer(sparse=False).fit_transform([x, y])
    #     mnk = nth_root(sum(pow(abs(a - b), 3) for a, b in vect.T), 3)
    #     if mnk:
    #         return 1/mnk
    #     else:
    #         return 1
    #
    # def get_cosine(s):
    #     try:
    #         vec1 = Counter(s['question1'].split(' '))
    #         vec2 = Counter(s['question2'].split(' '))
    #     except:
    #         return 0.0
    #     intersection = set(vec1.keys()) & set(vec2.keys())
    #     numerator = sum([vec1[x] * vec2[x] for x in intersection])
    #
    #     sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    #     sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    #     denominator = sqrt(sum1) * sqrt(sum2)
    #
    #     if not denominator:
    #         return 0.0
    #     else:
    #         return float(numerator) / denominator
    #
    # frame['cosine'] = frame[['question1', 'question2']].apply(get_cosine, axis=1)
    # frame['muta'] = frame[['question1', 'question2']].apply(my_mutalist, axis=1)
    # frame['mnk'] = frame[['question1', 'question2']].apply(my_mutamnk, axis=1)
    # frame['mnk'] = frame['mnk'].astype('float')
    #
    # E.saver(frame, 'all_and_my_mutalist.csv')

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

    print('                Cosine:', roc_auc_score(frame['is_duplicate'], frame['cosine'].fillna(0)))
    print('                  Muta:', roc_auc_score(frame['is_duplicate'], frame['muta'].fillna(0)))
    print('                   Mnk:', roc_auc_score(frame['is_duplicate'], frame['mnk'].fillna(0)))

    print('zunigrams_common_ratio:', roc_auc_score(frame['is_duplicate'], frame['zunigrams_common_ratio'].fillna(0)))
    print(' zbigrams_common_ratio:', roc_auc_score(frame['is_duplicate'], frame['zbigrams_common_ratio'].fillna(0)))
    print('zunigrams_common_count:', roc_auc_score(frame['is_duplicate'], frame['zunigrams_common_count'].fillna(0)))
    print(' zbigrams_common_count:', roc_auc_score(frame['is_duplicate'], frame['zbigrams_common_count'].fillna(0)))

    for i in ls:
        print(' %s:'%i, roc_auc_score(frame['is_duplicate'], frame[i].fillna(0)))

    x_train = pandas.DataFrame()
    x_test = pandas.DataFrame()
    del x_test
    x_train['rake'] = frame['rake']
    x_train['tfidf'] = frame['tfidf']
    x_train['dist'] = frame['dist']
    x_train['comp'] = frame['comp']
    x_train['diff'] = frame['diff']
    x_train['cosine'] = frame['cosine']
    x_train['muta'] = frame['muta']
    x_train['mnk'] = frame['mnk']

    x_train[ls] = frame[ls]

    x_train['zunigrams_common_ratio'] = frame['zunigrams_common_ratio']
    x_train['zbigrams_common_ratio'] = frame['zbigrams_common_ratio']
    x_train['zunigrams_common_count'] = frame['zunigrams_common_count']
    x_train['zbigrams_common_count'] = frame['zbigrams_common_count']

    y_train = frame['is_duplicate'].values

    del frame

    pos_train = x_train[y_train == 1]
    neg_train = x_train[y_train == 0]

    # Oversampling
    p = 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pandas.concat([neg_train, neg_train])
        scale -= 1
    neg_train = pandas.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    print(len(pos_train) / (len(pos_train) + len(neg_train)))

    x_train = pandas.concat([pos_train, neg_train])
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
    del pos_train, neg_train

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    del x_valid, y_valid

    x_train['is_duplicate'] = y_train
    del y_train

    print(Learning(x_train[['comp',
                            'rake',
                            'dist',
                            'diff',
                            'tfidf',
                            'cosine',
                            'muta',
                            'mnk',
                            'zbigrams_common_count',
                            'zunigrams_common_count',
                            'zbigrams_common_ratio',
                            'zunigrams_common_ratio',
                            'is_duplicate']+ls], y_col='is_duplicate').trees(m_params={
        'verbose': True,
        'criterion': 'mse',
        'n_estimators': 2500,
        'learning_rate': 0.07,
    },
        models=en.GradientBoostingClassifier))

if __name__ == '__main__':
    go()
