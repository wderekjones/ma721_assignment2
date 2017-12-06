# built using example code from pytorch getting started with examples documentation http://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# -*- coding: utf-8 -*-
import argparse
import time

import numpy as np

seed = 0
np.random.RandomState(seed)
import tqdm
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
import torch

import multiprocessing
torch.manual_seed(seed)
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import init
from sklearn.metrics import accuracy_score
from pytorch.input_pipeline import imdbTrainDataset
from pytorch.model import CNN

# TODO: add class weighted validation metrics for testing? should be easy x)
# TODO: roc curves or confusion matrices on the testing data, the latter should be more straightforward
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to data")
parser.add_argument("--feats", type=str, help="path to features")
parser.add_argument("--epochs", type=int, help="number of training epochs", default=10)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("--batch_sz", type=int, help="batch size", default=500)
parser.add_argument("--null", type=str, help="path to null features")
parser.add_argument("--ncores", type=int, help="number of cores to use for multiprocessing of training", default=multiprocessing.cpu_count())
parser.add_argument("--oversample", type=str, help="whether to oversample the minority class", default=None)
parser.add_argument("--p", type=float,help="dropout probability", default=0.5)
parser.add_argument("--model", type=str, help="name of model to use for training", default="Net")


def train(model, dataloader, optimizer):
    losses = []
    train_precisions = []
    train_f1s = []
    train_recalls = []
    accs = []

    start_train_clock = time.clock()
    for batch_number, batch in tqdm.tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        # there is no point in wrapping these in variables containing
        batch_xs = Variable(batch[0].cuda().long(), requires_grad=False)
        batch_ys = Variable(batch[1].cuda().long(), requires_grad=False)

        # Forward pass: compute output of the network by passing x through the model.
        y_pred_probs = model(batch_xs)

        y_pred = np.argmax(y_pred_probs.cpu().data.numpy(),axis=1)
        y_test = batch_ys.cpu().data.numpy()
        accs.append((accuracy_score(y_test,y_pred)))
        # Compute loss.
        loss = loss_fn(y_pred_probs.view(-1), batch_ys.float())
        losses.append(loss.cpu().data.numpy())

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward(retain_graph=True)
    optimizer.step()
    stop_train_clock = time.clock()
    # print("epoch time: {0:.3f}".format(stop_train_clock-start_train_clock))
    print("loss: {0:.3f} \t accuracy: {0:.3f}".format(np.mean(losses), np.mean(accs)))
    # witch the model to evaluation mode (training=False) in order to evaluate on the validation set

if __name__ == '__main__':
    args = parser.parse_args()
    time_stamp = time.time()

    num_epochs = args.epochs
    batch_size = args.batch_sz
    learning_rate = args.lr
    n_bins = 2
    num_workers = multiprocessing.cpu_count()

    # load the training data, then further partition into training and validation sets, preserving the ratio of
    # positives to negative training examples
    train_data = imdbTrainDataset()
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)


    # load the model
    model = CNN(vocab_size=20000, embedding_dim=128, hidden_dim=50, label_size=1, batch_size=batch_size, seq_len=250)
    model.cuda()

    # model._parameters = init.xavier_normal(list(model.parameters()))
    # or
    for param in model.parameters():
        # init.xavier_normal(param)
        init.uniform(param)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_name = "bce"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    start_clock = time.clock()
    epoch = 0

    print("time_stamp: {}".format(time_stamp))
    print()
    print("model: {}".format(args.model))
    print("data: {}".format(args.data))
    print("features: {}".format(args.feats))
    print("null features: {}".format(args.null))
    print("ncores: {}".format(args.ncores))
    print("oversample: {}".format(args.oversample))
    print("epochs: {}".format(args.epochs))


    # print("training data #examples: {} \t validation #examples: {}".format(len(train_idxs),len(val_idxs)))


    # output optimization details
    regularization = "dropout: drop_prob = "+str(args.p)
    initialization = "uniform"

    print("optimizer: {} \t lr: {} \t initialization: {} \t regularization: {}".format("SGD", args.lr, initialization,
                                                                                       regularization))
    #
    # train model
    #
    #

    for step in range(num_epochs):
        epoch_loss = 0

        train(model, train_dataloader, optimizer)
        epoch += 1

    stop_clock = time.clock()
    print()
    print("Train time: ", (stop_clock-start_clock), " cpu seconds.")
    print()