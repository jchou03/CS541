### My implementationfor the challenge

from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import tree, ensemble

'''
Notes about problem:
- regression problem
- train on input data, need to predict based on features
- evaluated on MSE of LOG prices

things to try:
- regression problem
  - linear regression
  - logistic regression
- neural network approaches -> pytorch
  - linear models + activation functions
  - residual connections?
- regression trees? -> sklearn
- boosting?
- 

what doesn't make sense:
- recurrent neural networks (not sequential data, so it doesn't make a ton of sense)
- convolutional NNs (the data comes from different features, different features can't necessarily be connected together)

approaches tried:
- sklearn.tree.DecisionTreeRegressor
  - train log MSE: 0.0417093276433341
  - validation log MSE: 0.3586
  - notes: overfitting to training data!!
      this is likely a naive approach to solving this problem (overfitting), let's try to do cross validation
    on the training data to gain a better understanding of how each model is performing and how much it is 
    overfitting


- sklearn.ensemble.GradientBoostingRegressor
  - train log MSE: 0.03463397524760063
  - validation log MSE: 
  - cross validation scores: [-0.13925216 -0.14871137 -0.13613333 -0.1396184  -0.14120412]
  - notes: trying to use gradient boosting to get things done

'''

class Model:
    # Modify your model, default is a linear regression model with random weights

    def __init__(self):
        self.model = ensemble.GradientBoostingRegressor()

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Train model with training data.
        Currently, we use a linear regression with random weights
        You need to modify this function.
        :param X_train: shape (N,d)
        :param y_train: shape (N,1)
            where N is the number of observations, d is feature dimension
        :return: None
        """
        N, d = X_train.shape

        self.model.fit(X_train, y_train)
        return None

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Use the trained model to predict on un-seen dataset
        You need to modify this function
        :param X_test: shape (N, d), where N is the number of observations, d is feature dimension
        return: prediction, shape (N,1)
        """
        y_pred = self.model.predict(X_test)
        return y_pred