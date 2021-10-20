import torch
import urllib.request
import os

from scipy.io import loadmat
from math import floor
from utils.deepGP import DeepGP
from utils.train import train
from utils.test import test

if __name__ == "__main__":
    # this is for running the notebook in our testing framework
    smoke_test = ('CI' in os.environ)


    if not smoke_test and not os.path.isfile('../elevators.mat'):
        print('Downloading \'elevators\' UCI dataset...')
        urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', './data/elevators.mat')


    if smoke_test:  # this is for running the notebook in our testing framework
        X, y = torch.randn(1000, 3), torch.randn(1000)
    else:
        data = torch.Tensor(loadmat('./data/elevators.mat')['data'])
        X = data[:, :-1]
        X = X - X.min(0)[0]
        X = 2 * (X / X.max(0)[0]) - 1
        y = data[:, -1]


    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()

    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

    num_epochs = 1 if smoke_test else 10
    num_samples = 3 if smoke_test else 10
    num_output_dims = 2 if smoke_test else 10

    model = DeepGP(train_x.shape,num_output_dims=num_output_dims)
    if torch.cuda.is_available():
        model = model.cuda()

    print("Starting training...")
    model = train(model, train_x, train_y, num_epochs = num_epochs, num_samples = num_samples)

    # validate
    rmse,nll = test(model,test_x, test_y, batch_size = 1)