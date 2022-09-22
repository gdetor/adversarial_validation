import numpy as np

import csv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import xgboost as xgb


class AdversarialValidation:
    def __init__(self,
                 n_train_examples=1000,
                 method='logreg'):
        self.n_train_examples = 1000
        self.method = method

    def prepareData(self, X_train, X_test):
        """!
        @param X_train Train data set (ndarray) of shape
        (n_samples, n_features)
        @param X_test Test data set (ndarray) of shape (n_samples, n_features)
        """

        if X_train.shape[1] != X_test.shape[1]:
            print("ERROR: The features dimension of train and test data sets\
                    does not match!")
            exit(-1)

        m = len(X_train) + len(X_test)
        Y = np.concatenate([X_train, X_test], axis=0)
        L = np.zeros((m, ), dtype='i')
        L[:len(X_train)] = 1

        tmp_data = train_test_split(Y,
                                    L,
                                    train_size=self.n_train_examples,
                                    shuffle=True)
        self.x_train, self.y_train = tmp_data[0], tmp_data[2]
        self.x_test, self.y_test = tmp_data[1], tmp_data[3]

    def getAUCScore(self, X_train, X_test):
        self.prepareData(X_train, X_test)

        if self.method == 'logreg':
            lr = LogisticRegression()
            lr.fit(self.x_train, self.y_train)
            prob = lr.predict_proba(self.x_test)[:, 1]
            AUC = roc_auc_score(self.y_test, prob)
        elif self.method == 'xgboost':
            XGBC = xgb.XGBClassifier(objective="binary:logistic",
                                     eval_metric="logloss",
                                     use_label_encoder=False,
                                     learning_rate=0.05,
                                     max_depth=5)
            XGBC.fit(self.x_train, self.y_train)
            prob = XGBC.predict_proba(self.x_test)[:, 1]
            AUC = roc_auc_score(self.y_test, prob)
        elif self.method == 'cv':
            XGBMat = xgb.DMatrix(data=self.x_train,
                                 label=self.y_train)
            params = {"objective": "binary:logistic",
                      "eval_metric": "logloss",
                      "learning_rate": 0.05,
                      "max_depth": 5,
                      }
            tmp_res = xgb.cv(params=params,
                             dtrain=XGBMat,
                             metrics="auc",
                             nfold=5,
                             num_boost_round=200,
                             early_stopping_rounds=20,
                             as_pandas=False)
            AUC = tmp_res['test-auc-mean'][0]
        else:
            print("ERROR: Classifier do not found!")
            print("Available methods: logreg, xgboost")
            exit(-1)
        return AUC


def readCSVFile(fname, delim=","):
    with open(fname, "r") as f:
        lines = csv.reader(f, delimiter=delim)
        data = list(lines)
    return np.array(data)


if __name__ == '__main__':
    import pandas as pd

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train = train.select_dtypes(include=['number']).copy()
    train = train.drop(['Survived'], axis=1)
    test = test.select_dtypes(include=['number']).copy()

    train = np.nan_to_num(train.values, 0)
    test = np.nan_to_num(test.values, 0)

    AV = AdversarialValidation(n_train_examples=1000, method='logreg')
    print(AV.getAUCScore(train, test))
