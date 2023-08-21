# Adversarial Validation

This repository contains a simple adversarial validation (AV) method implementation.
AV is a helpful method for investigating whether the train and test data sets come
from the same distribution. It was originally proposed in [1] and re-implemented in [2].

The identically distributed data assumption is violated when train and test data sets
are disjoint or *almost* disjoint (meaning they are not similar or drawn from the same distribution).
Thus a machine learning algorithm has serious troubles once deployed because it will
be used on data that has yet to be seen, so its performance will be bad. Therefore,
if we know *a priori* that this might happen, we can take appropriate action to prevent
it. AV method does precisely that. It mingles the train and test data sets and creates
one hyper set. It annotates each sample from the train set with label one (1) and any
other sample with label zero (0). Now, if a binary classifier cannot distinguish the
data and thus has a <0.5 AUC ROC score, we know that our data sets are fine and can
proceed. Otherwise, we have to remedy the data.



### Brief Description

In this repository we implement a Python class of the AV method. The AV class 
provides three binary clasiffiers (logistic regression, XGBoost classfier, 
and XGBoost cross-validation). Moreover, AV class implements seven different 
transforms that the user can apply on the raw data. More precisely, AV comes 
with the following transforms:
  * MinMaxScaler
  * StandardScaler
  * MaxAbsScaler
  * RobustScaler
  * L1Normalizer
  * L2Normalizer
  * MaxNormalizer

Furthermore, the user can apply a polynomial and interactions features 
transform on the raw data. 


### Example

This is an example of how to instantiate the AdversarialValidation class and
how to compute the AUC ROC score. 

```python
import numpy as np

from av import AdversarialValidation

train = ... # load the train data set you'd like to examine
test = ...  # load the test data set you'd like to examime

# instantiate the AdversartialValidation class
AV = AdversarialValidation(n_train_examples=1000, method='logreg')

# Compute the AUC score on the raw data
print(AV.getAUCScore(train, test))

# Reset AV class attributes
AV.reset()

# Transform the raw data and then compute the AUC score
print(AV.transformGetAUCScore(train,
                              test,
                              transform='StandardScaler',
                              polyFeatures=False))

```

### Dependencies

AdversarialValidation class requires the following Python packages:
  - Numpy
  - Sklearn
  - XGBoost


### References
  1. http://fastml.com/adversarial-validation-part-one/ 
  2. https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation/notebook

