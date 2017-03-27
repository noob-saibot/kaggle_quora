import pandas
from numpy import sqrt, log1p, expm1, log
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import itertools


class Extractor:
    def __init__(self, work_dir, file_train, file_test=None):
        self.work_dir = work_dir
        self.file_tr = file_train
        self.file_ts = file_test
        self.frame = None

    def __str__(self):
        return str(self.frame)

    def frame_corr(self, *data_frames, delta_lvl=None):
        """
        Correlation of elements of frame
        Parameters
        ----------
        :param data_frames: list
            List of frames.

        :param delta_lvl: float
            Correlation level.

        Returns
        -------
        :return _all_frames_corr: list
            List of correlation for each frame.

        :return self.frame: pandas.DataFrame
            Correlation for each element if MAIN frame
        """
        if data_frames:
            _all_frames_corr = []
            for index, data_frame in enumerate(data_frames):
                if isinstance(data_frame, pandas.DataFrame):
                    if delta_lvl:
                        _all_frames_corr.append(self.delta_corr(data_frame.corr(), delta_lvl))
                    else:
                        _all_frames_corr.append(data_frame.corr())
                else:
                    print(index, "It's not a data frame", sep=' --- ')
                    _all_frames_corr.append(None)
            return _all_frames_corr

        elif isinstance(self.frame, pandas.DataFrame):
            if delta_lvl:
                return self.delta_corr(self.frame.corr(), delta_lvl)
            else:
                return self.frame.corr()
        else:
            return 'Frame is empty.'

    # Frame creation
    def df_creation(self):
        self.frame = pandas.read_csv(self.work_dir+self.file_tr, error_bad_lines=False)
        if self.file_ts:
            frame_test = pandas.read_csv(self.work_dir + self.file_ts, error_bad_lines=False)
            return self.frame, frame_test
        return self.frame

    @staticmethod
    def delta_corr(data_frame, delta):
        """
        Correlation of elements of frame
        Parameters
        ----------
        :param data_frame: pandas.DataFrame
            Data frame.

        :param delta: list
            Correlation level.

        Returns
        -------
        :return data_frame: pandas.DataFrame
            Data frame with Nan values on delta places.
        """
        if isinstance(data_frame, pandas.DataFrame):
            for a in data_frame.columns:
                data_frame.ix[abs(data_frame[a]) < delta, a] = None
                data_frame.ix[abs(data_frame[a]) == 1.0, a] = None
            return data_frame

    @staticmethod
    def importance(data_frame, y_field, m_param, method=ExtraTreesRegressor):
        """
            Feature importance extraction
            Parameters
            ----------
            :param data_frame: pandas.DataFrame
                Data frame.

            :param y_field: str
                Name of column with respect to which we will extract feature importance.

            :param method: Class regressor of sklearn
                Regressor for feature extraction.

            :param m_param: tuple
                Tuple of parameters for Regressor.

            Returns
            -------
            :return feature_importances_: numpy.array
                Importance of features.

            :return columns: pandas.indexes.base.Index
                Names of features.
            """
        data_frame = data_frame.fillna(data_frame.mean())
        clf = method(**m_param).fit(data_frame.drop([y_field], axis=1), data_frame[y_field])
        return clf.feature_importances_, data_frame.drop([y_field], axis=1).columns

    @staticmethod
    def encoding_for_labels(data_frame):
        enc = LabelEncoder()
        for column in data_frame.select_dtypes(include=['object']).columns:
            data_frame[column] = data_frame[column].factorize()[0]
            data_frame[column] = enc.fit_transform(data_frame[column])
        return data_frame

    @staticmethod
    def normalize_it(n_frame, n_method=log1p):
        for col in n_frame:
            if n_frame[col].dtype != 'object':
                n_frame[col] = n_method(n_frame[col])
        return n_frame

    def saver(self, frame_save, name):
        frame_save.to_csv(self.work_dir+name)


class Viewer:
    import matplotlib
    matplotlib.style.use('ggplot')

    def __init__(self, data_frame):
        self.data_frame = data_frame.dropna(axis=0)

    def bar(self):
        self.data_frame.plot(kind='bar')
        plt.show()

    def line(self):
        self.data_frame.plot()
        plt.show()

    def site_chart(self):
        template = """{{labels: {labels}, datasets:
        [{{label: "Dataset", backgroundColor: window.chartColors.red,
        borderColor: window.chartColors.red, data: {data},
        fill: false,}}, {{}}]}},"""

        result_dict = {'labels': [x for x in self.data_frame.index.values],
                       'data': [x[0] for x in self.data_frame.values]}

        return template.format(**result_dict).replace('\'', '"')


class Learning:
    def __init__(self, data_frame, cross_params=5, y_col=''):
        self.data_frame = data_frame
        self.cross = cross_params
        if not y_col:
            self.slice = data_frame.columns[-1]
        else:
            self.slice = y_col

    def __str__(self):
        return str(self.data_frame)

    def folding(self):
        folds = KFold(n_splits=self.cross, shuffle=False)
        folds = folds.split(self.data_frame.drop(['SalePrice'], axis=1), self.data_frame['SalePrice'])
        # for train_index, test_index in folds:
        #     print(self.data_frame.shape)
        #     print("TRAIN:", len(train_index), "TEST:", len(test_index))
        return folds

    @staticmethod
    def root_mse_score(predictions, targets):
        return sqrt(((predictions - targets) ** 2).mean())

    def trees(self, m_params, models, cross=True, jobs=1):
        frame_l = self.data_frame.fillna(self.data_frame.mean())
        reg = models(**m_params)

        if cross:
            results = -cross_val_score(reg, frame_l.drop([self.slice], axis=1),
                                      frame_l[self.slice], cv=6, n_jobs=jobs,
                                      scoring='neg_log_loss',
                                      #fit_params={'verbose': True},
                                            )
            return results.mean()
        else:
            reg.fit(frame_l.drop([self.slice], axis=1), frame_l[self.slice])
            return reg


# if __name__ == "__main__":
#     drop_list = [
#     ]
#
#     E = Extractor(work_dir='./',
#                   file_train='data/train.csv',
#                   file_test='data/test.csv')
#     frame, frame2 = E.df_creation()
#
#     result_frame = pandas.concat([frame, frame2], axis=0)
#     result_frame = result_frame.drop(drop_list, axis=1)
#
#     import deep_analit
#
#     result_frame = deep_analit.go_deeper()
#
#     result_frame = filling_up.Filling().fill()
#
#     result_frame = pandas.get_dummies(result_frame)
#     result_frame = E.normalize_it(result_frame)
#
#     frame_train = result_frame.dropna(subset=['SalePrice'])
#     frame_test = result_frame[result_frame['SalePrice'].isnull()]
#
#     frame_train = frame_train.fillna(frame_train.mean())
#     X_train, X_test, y_train, y_test = train_test_split(frame_train.drop(['SalePrice'], axis=1),
#                                                         frame_train['SalePrice'],
#                                                         train_size=0.99)
#     #model = ExtraTreesRegressor(**dict_of_params)
#
#     dict_of_params = {
#         'criterion': 'mse',
#         'n_estimators': hp.choice('n_estimators', range(1000, 10000, 100)),
#         'learning_rate': hp.choice('learning_rate', [i/1000 for i in range(8, 35)]),
#         'loss': 'huber',
#         'max_features': 'sqrt',
#         'verbose': False,
#         #'n_estimators': 8090,
#         #'learning_rate': 0.023,
#         'subsample': hp.choice('subsample', [i/10 for i in range(3, 9)]),
#         'alpha': hp.choice('alpha', [i/10 for i in range(3, 9)])
#     }
#
#     model = BaggingRegressor
#
#     # print(Learning(frame_train, y_col='SalePrice').trees(m_params={}, models=LassoCV, cross=True))
#     # print(Learning(frame_train, y_col='SalePrice').trees(m_params={
#     #     'base_estimator': GradientBoostingRegressor(**dict_of_params),
#     #     'n_estimators': 4
#     # }, models=model))
#     #
#     # print(Learning(frame_train, y_col='SalePrice').trees(m_params={
#     #     'C': 100,
#     #     'epsilon': 0.1,
#     #     'verbose': True,
#     #     'kernel': 'sigmoid'
#     # }, models=SVR))
#     #
#     # exit()
#
#
#     def f(params):
#         #drop_data = params['drop_data']
#         #params.pop('drop_data', None)
#     #     acc = Learning(frame_train, y_col='SalePrice').trees(m_params={
#     #     'base_estimator': GradientBoostingRegressor(**params),
#     #     'n_estimators': 4
#     # }, models=model)
#         acc = Learning(frame_train, y_col='SalePrice').trees(m_params={
#             'n_estimators': 9,
#             'base_estimator': GradientBoostingRegressor(**params)
#         }, models=BaggingRegressor)
#         print(params, acc, sep='\n', end='\n\n')
#         print(params, acc, sep='\n', end='\n\n', file=open("opt_log_deep_deeper", 'a'))
#         return {'loss': -acc, 'status': STATUS_OK}
#
#     trials = Trials()
#     best = fmin(f, dict_of_params, algo=tpe.suggest, max_evals=1000, trials=trials)
#     print('best:', best)
#
#     exit()
#
#     model = model(n_estimators=16, base_estimator=GradientBoostingRegressor(**dict_of_params))
#
#     model.fit(X_train, y_train)
#
#     print(Learning.root_mse_score(model.predict(X_test), y_test))
#
#     submission = frame_test['SalePrice']
#     frame_test = frame_test.drop(['SalePrice'], axis=1)
#     frame_test = frame_test.fillna(frame_test.mean())
#     rs = expm1(model.predict(frame_test))
#
#     submission = pandas.DataFrame(data=rs, index=submission.index, columns=['SalePrice'])
#     E.saver(submission, 'sub.csv')




