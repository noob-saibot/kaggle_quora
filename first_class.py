from pandas import read_csv, DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import csv
import sklearn.preprocessing as pr
from sklearn.calibration import calibration_curve
import lasagne
import theano
import theano.tensor as T
from os import getcwd, listdir
import re
import time
import pickle
from sklearn.metrics import mean_absolute_error
# from xgboost import XGBRegressor
# import xgboost as xgb


class Fworker():
    def __init__(self, path = getcwd()+'/'):
        self.path = path

    def finder(self, name):
        for file in listdir(self.path + 'eval'):
            if name in file:
                print(file)
                return file

    def frame_gen(self, addr, index_col='id'):
        if index_col != 'id':
            return read_csv(self.path + '/eval/' + addr, ',')
        return read_csv(self.path + '/eval/' + addr, ',', index_col=index_col)

    def frame_sv(self, dataf, name):
        dataf.to_csv(self.path + '/eval/' + name, index=False)

    def frame_norm(self, df):
        for i in df.head(0):
            if min(df[i]) != 0:
                df[i] /= min(df[i])
            df[i] /= max(df[i])
        return df


class Proc():
    def __init__(self):
        pass

    def f_imp(self, X, y, method=ExtraTreesRegressor, m_param=[], save=False):
        clf = method(*m_param).fit(X, y)
        if save:
            with open(save, 'w') as f:
                for i, j in zip(X.columns.values, clf.feature_importances_):
                    f.write(str(i)+' ' + str(j) + '\n')
        return clf.feature_importances_

    def f_trainer(self, X, y, method=ExtraTreesRegressor, m_param=[], save='Demo_model'):
        list_params = tuple(i for i in m_param)
        string = '{0}{1}.fit({2}, {3})'.format('method', list_params, 'X', 'y').replace('\'', '')
        clf = eval(string)
        if save:
            pickle.dump(clf, open(save, "wb"))
        return clf

    def f_hist(self, *args):
        with open('log_tests', 'a') as f:
            for i in args:
                f.write(str(time.strftime("%a, %d %b %Y %H:%M:%S +0000 ", time.gmtime())))
                f.write(str(i)+'\n')

    def evalerror(self, preds, dtrain):
        labels = dtrain.get_label()
        return 'mae', mean_absolute_error(preds, labels)

    def build_mlp(self, input_var=None, ln=None, n_hp=10, n_ls=2, f_ac='tanh', out_f='linear', in_f='linear'):
        l_in = lasagne.layers.InputLayer(shape=(None, ln), input_var=input_var,
                                         nonlinearity=getattr(lasagne.nonlinearities, out_f))
        sp = ['l_in']+['l_hid%s'%s for s in range(1, n_ls+1)]
        for layer in sp[1:]:
                locals()[layer] = lasagne.layers.DenseLayer(
                                locals()[sp[sp.index(layer)-1]], num_units=n_hp,
                                nonlinearity=getattr(lasagne.nonlinearities, f_ac),
                                W=lasagne.init.Constant(0.5))
        l_out = lasagne.layers.DenseLayer(
            locals()['l_hid%s'%n_ls], num_units=1, nonlinearity=getattr(lasagne.nonlinearities, out_f))
        return l_out

    def norm(self, df):
        for i in df.head(0):
            if min(df[i]) != 0:
                df[i] /= min(df[i])
            df[i] /= max(df[i])
        return df

    def mae(self, preds, dtrain):
        return mean_absolute_error(preds, dtrain)

    def custom_objective(self, y_true, y_pred):
        return theano.tensor.sum(np.absolute((y_pred - y_true))) / y_true.shape[0]

if __name__ == "__main__":
    F = Fworker()
    addr = F.finder('train', )
    df_T = F.frame_gen(addr)

    df_T = pd.get_dummies(df_T.head(round(df_T.shape[0] / 1)))

    df_Y = df_T.loss
    #df_T = np.log1p(df_T.drop(['loss'], axis=1))
    df_T = df_T.drop(['loss'], axis=1)
    P = Proc()

    X_train, X_test, y_train, y_test = train_test_split(df_T, df_Y,
                                                        test_size=0.1)

    input_var_1 = T.dmatrix('input_var_1')
    input_var_2 = T.dmatrix('input_var_2')
    target_var = T.dmatrix('targets')

    df_CAT = X_train[[i for i in X_train.columns if i.startswith('cat')]]
    df_CONT = np.log1p(X_train[[i for i in X_train.columns if i.startswith('cont')]])
    df_CAT_t = X_test[[i for i in X_test.columns if i.startswith('cat')]]
    df_CONT_t = np.log1p(X_test[[i for i in X_test.columns if i.startswith('cont')]])

    l_in_1 = lasagne.layers.InputLayer(shape=(None, df_CAT.shape[1]), input_var=input_var_1,
                                     nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal())

    l_in_2 = lasagne.layers.InputLayer(shape=(None, df_CONT.shape[1]), input_var=input_var_2,
                                     nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotNormal())

    l_hid_1_1 = lasagne.layers.DenseLayer(
        l_in_1, num_units=12,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotNormal())

    l_hid_1_2 = lasagne.layers.DenseLayer(
        l_hid_1_1, num_units=12,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotNormal())

    l_hid_1_3 = lasagne.layers.DenseLayer(
        l_hid_1_2, num_units=12,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotNormal())

    l_hid_2_1 = lasagne.layers.DenseLayer(
        l_in_2, num_units=5,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotNormal())

    l_hid_2_2 = lasagne.layers.DenseLayer(
        l_hid_2_1, num_units=5,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotNormal())

    l_hid_2_1_1 = lasagne.layers.DenseLayer(
    l_in_1, num_units=14,
    nonlinearity=lasagne.nonlinearities.leaky_rectify,
    W=lasagne.init.GlorotNormal())

    l_hid_2_1_2 = lasagne.layers.DenseLayer(
    l_hid_2_1_1, num_units=14,
    nonlinearity=lasagne.nonlinearities.leaky_rectify,
    W=lasagne.init.GlorotNormal())

    l_hid_2_1_3 = lasagne.layers.DenseLayer(
        l_hid_2_1_2, num_units=14,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,
        W=lasagne.init.GlorotNormal())

    l_hid_2_2_1 = lasagne.layers.DenseLayer(
    l_in_2, num_units=8,
    nonlinearity=lasagne.nonlinearities.leaky_rectify,
    W=lasagne.init.GlorotNormal())

    l_hid_2_2_2 = lasagne.layers.DenseLayer(
    l_hid_2_2_1, num_units=8,
    nonlinearity=lasagne.nonlinearities.leaky_rectify,
    W=lasagne.init.GlorotNormal())

    l_merg = lasagne.layers.ConcatLayer([l_hid_1_3, l_hid_2_2, l_hid_2_1_3, l_hid_2_2_2])

    l_out = lasagne.layers.DenseLayer(
        l_merg, num_units=1, nonlinearity=lasagne.nonlinearities.linear)

    # network = P.build_mlp(input_var, X_train.shape[1], n_hp=4, f_ac='linear', out_f='very_leaky_rectify',
    #                       in_f='leaky_rectify', n_ls=3)

    network = l_out

    prediction = lasagne.layers.get_output(network, {l_in_1: input_var_1, l_in_2: input_var_2})

    # mse loss function
    # loss = lasagne.objectives.squared_error(prediction, target_var)

    loss = P.custom_objective(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=0.01)

    train_fn = theano.function([input_var_1, input_var_2, target_var], loss, updates=updates)
    test_loss = lasagne.objectives.get_output(network, deterministic=False)
    test_fn = theano.function([input_var_1, input_var_2], test_loss)
    num_epochs = 1000
    print(y_train.reshape(len(y_train), 1))

    y_train = y_train.reshape(len(y_train), 1)
    for epoch in range(num_epochs):
        inputs1, inputs2, targets = (df_CAT, df_CONT, y_train)
        print(epoch, train_fn(inputs1, inputs2, targets))
        if epoch % 100 == 0:
            print(mean_absolute_error(test_fn(df_CAT_t, df_CONT_t), y_test))

    y_test = y_test.reshape(len(y_test), 1)
    inputs1, inputs2, targets = (df_CAT_t, df_CONT_t, y_test)
    print(mean_absolute_error(test_fn(inputs1, inputs2), y_test))

