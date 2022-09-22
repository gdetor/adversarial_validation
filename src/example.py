# This is a demo script for how to use the AdversarialValidation class.
# Copyright (C) 2022  Georgios Is. Detorakis
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

from av import AdversarialValidation


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
    AV.reset()
    print(AV.transformGetAUCScore(train,
                                  test,
                                  transform='MinMaxScaler',
                                  polyFeatures=False))
