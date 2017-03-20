import lasagne
from data_extraction import Extractor, Learning
import pandas
from sklearn.model_selection import train_test_split
import theano.tensor as T
import theano
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import first_class
import numpy as np
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator, RegressorMixin

E = Extractor(work_dir='./',
              file_train='data/train.csv',
              file_test='data/sample_submission.csv')
frame, frame2 = E.df_creation()

result_frame = pandas.concat([frame, frame2], axis=1)

E = Extractor(work_dir='./',
              file_train='data/test.csv')
frame = E.df_creation()

frame2 = pandas.DataFrame([np.nan] * frame2.shape[0], columns=['Result'])

frame = pandas.concat([frame, frame2], axis=1)

result_frame = pandas.concat([result_frame, frame], axis=0, ignore_index=True)

frame_train = result_frame.dropna(subset=['Result'])
tmp = result_frame['Result']
tmp2 = frame_train.drop(['Result'], axis=1)
frame_train = (tmp2 - tmp2.mean()) / (tmp2.max() - tmp2.min())
frame_train = frame_train.fillna(frame_train.mean())
frame_train['Result'] = tmp

frame_test = result_frame[result_frame['Result'].isnull()]
frame_test = (frame_test - frame_test.mean()) / (frame_test.max() - frame_test.min())
frame_test = frame_test.fillna(frame_test.mean())

X_train, X_test, y_train, y_test = train_test_split(frame_train.drop(['Result'], axis=1),
                                                    frame_train['Result'],
                                                    train_size=0.80)

input_var = T.dmatrix('inputs')
target_var = T.dmatrix('targets')


class CustomEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, classificators=None):
        self.classificators = classificators

    def fit(self, X, y):
        for classificator in self.classificators:
            classificator.fit(X, y)

    def predict_proba(self, X):
        self.predictions_ = list()
        for classificator in self.classificators:
            self.predictions_.append(classificator.predict_proba(X).ravel())

        return np.mean(self.predictions_, axis=0)


class CustomNN():
    def __init__(self, verbose=True):
        self.model = None
        self.verbose = verbose

    def fit(self, X_train, y_train):
        network = first_class.Proc().build_mlp(input_var,
                                               X_train.shape[1],
                                               n_hp=round(X_train.shape[1]*3),
                                               f_ac='tanh',
                                               out_f='linear',
                                               in_f='tanh',
                                               n_ls=3)
        prediction = lasagne.layers.get_output(network, input_var)

        loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        test_loss = lasagne.layers.get_output(network, deterministic=False)
        test_fn = theano.function([input_var], test_loss)
        num_epochs = 7000

        y_train = y_train.reshape(len(y_train), 1)
        for epoch in range(num_epochs):
            inputs, targets = (X_train, y_train)
            train_fn(inputs, targets)
            if self.verbose and epoch % 20 == 0:
                rs = test_fn(X_test)
                print(log_loss(y_test, rs.reshape(1, len(rs))[0]))
        self.model = network

    def predict_proba(self, X_test):
        test_loss = lasagne.layers.get_output(self.model, deterministic=False)
        test_fn = theano.function([input_var], test_loss)
        return test_fn(X_test)

#ANN = CustomNN()
#ANN.fit(X_train, y_train)
#print(log_loss(y_test, ANN.predict_proba(X_test)))

if __name__ == "__main__":
    # print(Learning(frame_train, y_col='Result').trees(
    #     m_params={'classificators': [CustomNN(verbose=False)]},
    #     models=CustomEnsembleRegressor,
    #     cross=True))
    model = CustomEnsembleRegressor([CustomNN(verbose=False)])
    model.fit(X_train, y_train)
    submission = frame_test['Result']
    frame_test = frame_test.drop(['Result'], axis=1)
    print(log_loss(y_test, model.predict_proba(X_test)))
    print(log_loss(y_train, model.predict_proba(X_train)))
    rs = model.predict_proba(frame_test)

    submission = pandas.DataFrame(data=rs, columns=['Result'])
    E.saver(submission, 'sub.csv')
#print(Learning(frame_train, y_col='Result').trees(m_params={}, models=CustomNN, cross=True))
# exit()
# y_test = y_test.reshape(len(y_test), 1)
# inputs, targets = (X_test, y_test)
# print(log_loss(y_test, test_fn(X_test)))