### My implementationfor the challenge

from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader, Dataset
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
  - validation log MSE: 0.1674
  - cross validation scores: [-0.13925216 -0.14871137 -0.13613333 -0.1396184  -0.14120412]
  - notes: trying to use gradient boosting to get things done

- linear feedforward neural networks with ReLU activation functions
  -
'''

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.fc1 = nn.Linear(764, 382)
    self.fc2 = nn.Linear(382, 100)
    self.fc3 = nn.Linear(100, 1)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    return x

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
    self.X = torch.tensor(X.values, dtype=torch.float32)
    self.y = torch.tensor(y.values, dtype=torch.float32)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

class Model:
    def __init__(self):
        self.model = MyModel()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

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

        # convert data into tensors
        data_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        target_tensor = torch.tensor(y_train.values, dtype=torch.float32)

        # convert data to dataloader
        dataset = MyDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # set model to train
        self.model.train()
        # epochs
        for epoch in range(10):
          running_loss = 0.
          last_loss = 0.
          for i, (images, labels) in enumerate(dataloader):
            # forward pass
            outputs = self.model(images)
            loss = self.loss(outputs, labels)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print loss
            running_loss += loss.item()
            if i % 100 == 99:
              last_loss = running_loss / 1000 # loss per batch
              print('  batch {} loss: {}'.format(i + 1, last_loss))
              # tb_x = i * len(dataloader) + i + 1
              # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
              # running_loss = 0.

        return None

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Use the trained model to predict on un-seen dataset
        You need to modify this function
        :param X_test: shape (N, d), where N is the number of observations, d is feature dimension
        return: prediction, shape (N,1)
        """
        # first covert x_test to tensors
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

        # set model to eval
        self.model.eval()

        y_pred = self.model(X_test_tensor)
        return y_pred