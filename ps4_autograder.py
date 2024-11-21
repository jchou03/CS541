## import some libraries
import sklearn
from sklearn import datasets
import numpy as np
from typing import Tuple, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from sklearn import tree
from tqdm import tqdm
import torch
from torch import Tensor
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from IPython.display import display
import random
import re
from collections import Counter
import random
import string
from hmmlearn import hmm
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
from time import time
from collections import defaultdict
import torch.nn.functional as F

def question_3_1(file_path: str) -> str:
    '''
    file_path: path to the txt file downloaded from https://www.gutenberg.org/cache/epub/74589/pg74589.txt
    Returns:
    - simplified_text: contains only lowercase letters and single spaces.
    '''
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Write your code in this block ----------------
    # lowercase text
    text = text.lower()
    # use regular expressions to condense whitespace
    simplified_text = re.sub(r'[1-9]', ' ', text)
    simplified_text = re.sub(r'[^a-z\s]', '', simplified_text)

    # line might not be needed since number of spaces match
    # simplified_text = ' '.join(re.findall(r'\w+', simplified_text))
    # print(simplified_text)
    simplified_text = re.sub(r'\s+', ' ', simplified_text)
    # End of your code ---------------------------
    return simplified_text

def question_3_2(text: str, n: int) -> Counter:
    '''
    - text: str, the input text to process
    - n: int, the length of the n-grams to count

    Returns:
    - Counter: a Counter object where keys are n-grams and values are their respective counts
    '''

    # Write your code in this block ----------------
    grams = {}
    for i in range(len(text)):
      if(i + n <= len(text)):
        gram = text[i:i+n]
        if gram in grams:
          grams[gram] += 1
        else:
          grams[gram] = 1


    # End of your code ---------------------------
    return Counter(grams)

def question_3_3(ngram_counts: dict,  vocab_size: int) -> dict:
    '''
    - ngram_counts: dict, a dictionary containing n-gram counts
    - total_ngrams: int, the total number of n-grams in the text
    - vocab_size: int, the size of the vocabulary (unique n-grams)

    Returns:
    - ngram_probs: dict, a dictionary with n-grams as keys and their smoothed probabilities as values
    '''

    # Write your code in this block ----------------
    total_ngrams = 0
    for ngram in ngram_counts:
      total_ngrams += ngram_counts[ngram]
    print(total_ngrams)

    ngram_probs = {}

    for ngram in ngram_counts:
      ngram_probs[ngram] = ngram_counts[ngram] / total_ngrams

    # End of your code ---------------------------
    return ngram_probs

def question_3_4(text: str, pc: float, vocab: list, seed: int) -> str:
    '''
    Arguments:
    - text: str, the input text to be corrupted
    - pc: float, the probability of replacing each character
    - vocab: list, the vocabulary of characters to choose from when replacing text

    Returns:
    - corrupted_text:the corrupted version of the input text
    '''

    # Write your code in this block ----------------
    res = ""
    # Set the random seed for reproducibility
    random.seed(seed)
    for c in text:
      if random.random() < pc:
        res += random.choice(vocab)
      else:
        res += c

    return res
    # End of your code ---------------------------

def question3_5(bigram_counts: dict, vocab: list) -> np.ndarray:
    '''
    Arguments:
    - bigram_counts: dict, a dictionary where keys are bigrams and values are the counts of those bigrams
    - vocab:a list of all possible characters

    Returns:
    - transition_matrix: shape (vocab_size, vocab_size) where each entry represents the
                          probability of transitioning from one character to another
    '''

    # Write your code in this block ----------------
    # Initialize transition matrix
    transition_matrix = np.zeros((len(vocab), len(vocab)))

    # Calculate total bigram counts for each character
    char_bigram_counts = {}

    for bigram in bigram_counts:
      a = bigram[0]
      if a in char_bigram_counts:
        char_bigram_counts[a] += bigram_counts[bigram]
      else:
        char_bigram_counts[a] = bigram_counts[bigram]

    # Fill transition matrix with bigram probabilities
    for bigram in bigram_counts:
      a = bigram[0]
      b = bigram[1]

      a_index = vocab.index(a)
      b_index = vocab.index(b)

      transition_matrix[a_index, b_index] = bigram_counts[bigram] / char_bigram_counts[a]


    # End of your code ---------------------------
    return transition_matrix

def question3_6(vocab_size: int, corruption_prob: float) -> np.ndarray:
    '''
    Arguments:
    - vocab_size: int, the number of characters in the vocabulary
    - corruption_prob: float, the probability of a character being corrupted

    Returns:
    - emission_matrix: shape (vocab_size, vocab_size) where each entry represents the probability of observing one character given the hidden character
    '''

    # Write your code in this block ----------------

    emission_matrix = np.ones((vocab_size, vocab_size)) * (corruption_prob / (vocab_size - 1))
    np.fill_diagonal(emission_matrix, 1 - corruption_prob)


    # End of your code ---------------------------
    return emission_matrix

def question3_7(text: str, vocab: list) -> list:
    '''
    Arguments:
    - text: str, the input text to be converted into indices
    - vocab: list, the list of characters in the vocabulary

    Returns:
    - indices: a list of integers where each integer is the index of the corresponding character in the vocabulary
    '''

    # Write your code in this block ----------------
    indices = [vocab.index(c) for c in text]
    # End of your code ---------------------------
    return indices

def question3_8(indices: list, vocab: list) -> str:
    '''
    Arguments:
    - indices: list of integers, where each integer is an index in the vocabulary
    - vocab: list of characters in the vocabulary

    Returns:
    - A string where each character corresponds to the index from the vocabulary
    '''

    # Write your code in this block ----------------
    result = "".join([vocab[i] for i in indices])
    # End of your code ----------------------------

    return result

def question3_9(indices: list, vocab_size: int) -> np.ndarray:
    '''

    - indices: list of integers, where each integer is an index in the vocabulary
    - vocab_size: the size of the vocabulary

    Returns:
    - A 2D numpy array representing one-hot encoded vectors with shape (len(indices), vocab_size)
    '''

    # Write your code in this block ----------------
    one_hot_encoded = np.zeros((len(indices), vocab_size))
    one_hot_encoded[np.arange(one_hot_encoded.shape[0]), indices] = 1
    # End of your code ----------------------------

    return one_hot_encoded

def question3_10(original_text: str,bigram_counts:dict, vocab: list, pc: float) -> hmm.MultinomialHMM:
    '''
    Arguments:
    - original_text: str, the original text from which to train the HMM
    - vocab: list, the list of characters in the vocabulary
    - pc: float, the probability of character corruption (used for the emission matrix)

    Returns:
    - model: hmm.MultinomialHMM, the trained Hidden Markov Model
    '''

    # Write your code in this block ----------------
    # build the transion matrix and emission matrix. You can use question3_5 and question3_6
    transition_matrix = question3_5(bigram_counts, vocab)
    emission_matrix = question3_6(len(vocab), pc)


    # Initialize the HMM model using  hmm.MultinomialHMM and set n_trials to 1
    # Initialize with with uniform start probability
    model = hmm.MultinomialHMM(n_components=len(vocab), n_trials = 1)
    model.startprob_ = np.ones(len(vocab)) / len(vocab)
    model.transmat_ = transition_matrix
    model.emissionprob_ = emission_matrix

    # End of your code ---------------------------
    return model

def question3_11(original_text: str, recovered_text: str) -> float:
    '''
    - original_text: str, the original text
    - recovered_text: str, the recovered text

    Returns:
    - error_rate: float, the error rate as the proportion of differing characters
    '''
     # Write your code in this block ----------------
    total_chars = len(original_text)
    diff_chars = sum(1 for a, b in zip(original_text, recovered_text) if a != b)
    error_rate = diff_chars / total_chars
    # End of your code ---------------------------

    return error_rate

def question_4_2(transform: transforms.Compose, batch_size:int, shuffle:bool, drop_last:bool) -> DataLoader:
    """
    Similar to the example above, create then return a DataLoader for testset
    """
    # Write your code in this block -----------------------------------------------------------
    ## Step 1: create `testset` using datasets.CIFAR10: similar to `trainset` above, with
    # download=True, transform=transform but set `train` to False
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    ## Step 2: create `testloader` using DataLoader and passing params `batch_size`, `shuffer`, `drop_last`
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    ## Step 3: return `testloader`
    return testloader
    # End of your code -----------------------------------------------------------

class MyMLP(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        """
          in_dim: input dimension, usually we flatten 3d images  (num_channels, width, height) to 1d (num_channels * width * height),
                so we have in_dim = num_channels * width * height
          hid_dim: hidden dimension
          out_dim: output dimension
        """
        super().__init__()

        ## Complete the code below to initilaize self.linear_1, self.linear_2, self.relu
        # where self.linear_1 and self.linear_2 are `nn.Linear` objects, shape of (in_dim, hid_dim) and (hid_dim, out_dim) respectively,
        # and self.relu is a `nn.ReLU` object.
        # Write your code in this block -----------------------------------------------------------
        self.lc1 = nn.Linear(in_dim, hid_dim)
        self.lc2 = nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()
        # End of your code ------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        ## Assume we want to build a model as following: input `x` -> linear_1 -> relu -> linear_2
        ## Write your forward pass
        # Write your code in this block -----------------------------------------------------------
        x = self.lc1(x)
        x = self.relu(x)
        x = self.lc2(x)
        return x
        # End of your code ------------------------------------------------------------------------

def question_5_2_train_one_epoch(model: nn.Module, trainloader: DataLoader, device:torch.device,
                                 optimizer: torch.optim.SGD, criterion: torch.nn.CrossEntropyLoss, batch_size: int, flatten: bool):
    """
      Train 1 epoch on trainloader. You need to fill in after "##### [YOUR CODE]"
    """

    ## Set model to "train" model
    model = model.train()

    ## Keep track of loss and accuracy
    train_loss = 0.0
    train_acc = 0.0

    ## Loop over all the batches
    for i, (images, labels) in tqdm(enumerate(trainloader, 1), total=len(trainloader), desc=f"training 1 epoch..."):
        # For each batch, we have:
        #     + `images`: `bath_size` images in training set
        #     + `labels`: labels of the images (`batch_size` labels)


        ## Reshape the input dimension if we use MLP: instead of 3d (num_channels, width, height),
        # we flatten it to 1d (num_channels * width * height)
        if flatten:
            images = images.reshape(batch_size, -1)

        ## Move images and labels to `device` (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)

        # Write your code in this block -------------------------------------------------------------------------------------------
        ## We use 5 following steps to train a Pytorch model

        ##### [YOUR CODE] Step 1. Forward pass: pass the data forward, the model try its best to predict what the output should be
        # You need to get the output from the model, store in a new variable named `logits`
        logits = model(images)

        ##### [YOUR CODE] Step 2. Compare the output that the model gives us with the real labels
        ## You need to compute the loss, store in a new variable named `loss`
        loss = criterion(logits, labels)


        ##### [YOUR CODE] Step 3. Clear the gradient buffer
        optimizer.zero_grad()

        ##### [YOUR CODE] Step 4. Backward pass: calculate partial derivatives of the loss w.r.t parameters
        loss.backward()

        ##### [YOUR CODE] Step 5. Update the parameters by stepping in the opposite direction from the gradient
        optimizer.step()

        # End of your code --------------------------------------------------------------------------------------------------------
        ## Compute loss and accuracy for this batch
        train_loss += loss.detach().item()
        train_acc += question_5_3_compute_accuracy(logits, labels, batch_size)

    return train_loss/i, train_acc/i ## avg loss and acc over all batches

def question_5_3_compute_accuracy(logits: Tensor, labels: Tensor, batch_size: int) -> float:
    '''
      Obtain accuracy for a training batch
      logits: float Tensor, shape (batch_size, num_classes),  output from the model
      labels: Long Tensor, shape (batch_size, ), contains labels for the predictions
      batch_size: int, batch size

      Return accuracy for this batch, which should be a float number in [0, 100], NOT a Tensor
    '''

    # Write your code in this block ----------------
    _, predicted = torch.max(logits, 1)
    # print(predicted)
    correct = (predicted == labels).sum().item()
    accuracy = (correct / batch_size) * 100
    return accuracy
    # End of your code ---------------------------

## Note that we use `torch.no_grad()` here to disable gradient calculation.
# It will reduce memory consumption as we don't need to compute gradients in inference.

@torch.no_grad()
def question_5_4_evaluate(model: nn.Module, testloader: DataLoader, criterion, batch_size, device, flatten: bool):
    """
    You need to fill in after "##### [YOUR CODE]"
    """

    test_acc = 0.0
    test_loss = 0.0

    ## Turn on the evaluation mode
    model.eval()

    ## Loop through each batch on test set
    for i, (images, labels) in enumerate(testloader, 1):

        ## Flatten the image into 1d if using MLP
        if flatten:
            images = images.reshape(batch_size, -1)

        # Write your code in this block -----------------------------------------------------------

        ##### [YOUR CODE] Move data to `device`
        images = images.to(device)
        labels = labels.to(device)

        ##### [YOUR CODE] forward pass to get the output of the model
        logits = model(images)

        ##### [YOUR CODE] Compute the loss
        loss = criterion(logits, labels)
        test_loss += loss.detach().item()

        #### [YOUR CODE]  Compute accuracy (re-use question 4.3)
        test_acc += question_5_3_compute_accuracy(logits, labels, batch_size)

        # End of your code ---------------------------------------------------------------------------

    return test_loss/i, test_acc/i ## avg loss and acc over all batches

def question_5_5_train_model(model, device, num_epochs, batch_size, trainloader, testloader, flatten: bool = False):
    """
    model: Our neural net
    device: CPU/GPU
    num_epochs: How many epochs to train
    batch_size: batch size
    train/test loaders: training/test data
    flatten: whether we want to flatten the input image from 3d to 1d

    You need to fill in after "##### [YOUR CODE]"
    """

    # Write your code in this block -------------------------------------------------------------------------------------------

    ##### [YOUR CODE] create your optimizer using `optim.SGD`, set learning rate to 0.001, and momentum to 0.9
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    ##### [YOUR CODE] create criterion using `nn.CrossEntropyLoss`
    criterion = nn.CrossEntropyLoss()

    ## Measure runtime
    t_start = time()

    ## Store training log
    history = defaultdict(list)

    # We will train the model `num_epochs` times
    for i in range(1, num_epochs+1):
        ###### [YOUR CODE] train 1 epoch: call the function in question 4.2
        train_loss, train_acc = question_5_2_train_one_epoch(model, trainloader, device, optimizer, criterion, batch_size, flatten)

        ###### [YOUR CODE] call function in question 4.4 to see how it performs on test set
        test_loss, test_acc = question_5_4_evaluate(model, testloader, criterion, batch_size, device, flatten)
        # End of your code ----------------------------------------------------------------------------------------------------


        ## store train/test loss, accuracy
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        ## print out train/test loss, accuracy
        print(f'Epoch: {i} | Runtime: {((time()-t_start)/60):.2f}[m] | train_loss: {train_loss:.3f} | train_acc: {train_acc:.3f} | test_loss: {test_loss:.3f} | test_acc: {test_acc:.3f}')
    return history

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ## Recall that input images are 3x32x32,
        # i.e., 3 channels (red, green, blue), each of size 32x32 pixels.


        ## An example of conv and pooling layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)       ##### [YOUR CODE]
        # The first convolutional layer, `conv1`, expects 3 input channels,
        # and will convolve 32 filters each of size 3x5x5.
        # Since padding is set to 0 and stride is set to 1 as default,
        # the output size is (32, 28, 28).
        # This layer has ((3*5*5)+1)*32 = 2,432 parameters

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2) ##### [YOUR CODE]
        ## The first down-sampling layer uses max pooling with a (2,2) kernel
        # and a stride of 2. This effectively drops half of spatial size.

        self.conv2 = nn.Conv2d(32, 16, kernel_size=5) ##### [YOUR CODE]
        ## Similarly, we make another conv layer

        # Write your code in this block --------------------------------------------------------------

        self.fc1 = nn.Linear(400, 64) ##### [YOUR CODE]
        ## fc1 is a Linear layer. You'll need to look at the output of conv2 and the input of fc2 to
        # determine the in_dim and out_dim for fc1. You can do this by printing out the shape of the output of conv2 in forward() function.
        # End of your code ---------------------------------------------------------------------------


        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):

        # Implement your forward pass
        # Write your code in this block --------------------------------------------------------------
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        # End of your code ---------------------------------------------------------------------------

        return x
    

