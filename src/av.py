# This script implements the AdversarialValidation method for exploring if
# selected train/test data sets are of the same distribution.
# Copyright (C) 2022 Georgios Is. Detorakis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import MaxAbsScaler, Normalizer, PolynomialFeatures

import xgboost as xgb


class AdversarialValidation:
    """! This class implements the adversarial validation AV method that
    provides a means to test if the training/testing data sets are similar and
    the identically distributed assumption is not being violated. The class
    implements three methods: 1. prepareData that builds the hyper set that
    contains the original training and testing data sets. Moreover, creates
    the labels for the binary classification used by the AV method. 2.
    getAUCScore estimates the AUC ROC score that determines if the two
    (train/test) data sets are of the same distribution (i.e., the binary
    classifier cannot classify the data correctly - AUC score <0.5). 3.
    transformerGetAUCSCore applies some basic transforms on the original data
    and then it estimates the AUC ROC score.
    """
    def __init__(self,
                 n_train_examples=1000,
                 method='logreg'):
        """! Constructor method of AdversarialValidation class.

        @param n_train_examples Defines how many training samples will be used
        (int)
        @param method A string that determines which classification method will
        be used: 1. logreg - logistic regression, 2. xgboost - XGBoost, 3. cv -
        cross validation with xgboost.

        @return void
        """
        self.n_train_examples = 1000
        self.method = method

        self.reset()

    def reset(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def prepareData(self, X_train, X_test):
        """! Generates the data set for training/testing a binary classifier
        per AV method. In addition, it creates the necessary labels 1 - for
        training data 0 - for testing data.

        @param X_train Train data set (ndarray) of shape
        (n_samples, n_features)
        @param X_test Test data set (ndarray) of shape (n_samples, n_features)
        """

        if X_train.shape[1] != X_test.shape[1]:
            print("ERROR: The features dimension of train and test data sets\
                    does not match!")
            exit(-1)

        m = len(X_train) + len(X_test)  # total length of data set
        Y = np.concatenate([X_train, X_test], axis=0)   # data set
        L = np.zeros((m, ), dtype='i')                  # labels
        L[:len(X_train)] = 1

        # split train/test data sets
        tmp_data = train_test_split(Y,
                                    L,
                                    train_size=self.n_train_examples,
                                    shuffle=True)
        self.x_train, self.y_train = tmp_data[0], tmp_data[2]
        self.x_test, self.y_test = tmp_data[1], tmp_data[3]

    def getAUCScore(self, X_train, X_test):
        """! Estimates the AUC ROC score for the binary classification problem
        per AV method.

        @param X_train The original train data set of shape (n_samples,
        n_features)
        @param X_test The original test data set with of shape (n_samples,
        n_features)

        @return AUC the AUC-ROC score of our binary classifier.
        """
        # Prepare the data sets
        self.prepareData(X_train, X_test)

        # Perform a binary classification based on the chosen method
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

    def transformGetAUCScore(self,
                             X_train,
                             X_test,
                             transform='MinMaxScaler',
                             polyFeatures=False):
        """! Estimates the AUC ROC score for the binary classification problem
        per AV method after applying a transform on the data set.

        @note The available transforms are: 1. MinMaxScaler, 2. StandardScaler,
        3. MaxAbsScaler, 4. RobustScaler, 5. L1Normalizer, 6. L2Normalizer, 7.
        MaxNormalizer. Furthermore, this method can perform a polynomial and
        interactions features generation.

        @param X_train The original train data set of shape (n_samples,
        n_features)
        @param X_test The original test data set with of shape (n_samples,
        n_features)
        @param transform A string that determines which transform will be
        applied on the data set (see note above)
        @param polyFeatures A bool tha enables/disables the polynomial and
        interactions features generation

        @return AUC the AUC-ROC score of our binary classifier.
        """
        # Prepare the data sets
        self.prepareData(X_train, X_test)

        if transform == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif transform == 'StandardScaler':
            scaler = StandardScaler()
        elif transform == 'MaxAbsScaler':
            scaler = MaxAbsScaler()
        elif transform == 'RobustScaler':
            scaler = RobustScaler()
        elif transform == 'L1Normalizer':
            scaler = Normalizer(norm='l1')
        elif transform == 'L2Normalizer':
            scaler = Normalizer(norm='l2')
        elif transform == 'MaxNormalizer':
            scaler = Normalizer(norm='max')
        else:
            print("Transform not found! Choose one of the following:")
            print("MinMaxScaler")
            print("StandardScaler")
            print("RobustScaler")
            print("L1Normalizer")
            print("L2Normalizer")
            print("MaxNormalizer")
            exit(-1)

        if polyFeatures:
            scaler = Pipeline([('feats', PolynomialFeatures()),
                               ('scaler', scaler)])

        x_train_t = scaler.fit_transform(self.x_train)
        x_test_t = scaler.transform(self.x_test)
        AUC = self.getAUCScore(x_train_t, x_test_t)
        return AUC
