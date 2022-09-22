# adversarial_validation

This repository contains a simple implementation of the adversarial
validation (AV) method.  AV is a helpful method to investigate whether
the train and test data sets come from the same distribution.
It was originally proposed in [1] and re-implemented in [2].

When train and test data sets are disjoint or *almost* disjoint
(meaning they are not similar or coming from the same distribution),
the identically distributed data assumption is violated, and thus a
machine learning algorithm might have serious troubles once deployed.
This is because it will be deployed on data for the first time, so
its performance will be inferior.  Therefore, if we need *a priori*
that this might happen, we can take appropriate action to prevent it.
AV method does precisely that.  It mingles the train and test data
sets and creates one hyper set.  It annotates each sample from the
train set with label one and any other sample with label 0.  Now, if
a binary classifier cannot distinguish the data and thus has a <0.5
AUC ROC score, then we know that our data sets are fine, and we can
proceed. Otherwise, we have to remedy the data.



### Brief Description

In this repository we provide a class that implements the AV method. The AV class 
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

# Transform the raw data and then compute the AUC score
print(AV.transformGetAUCScore(transform='StandardScaler',
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

