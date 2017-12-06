import model
import input_pipeline
import torch



# built using example code from pytorch getting started with examples documentation http://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
import os
seed = 0
np.random.RandomState(seed)
import tqdm
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-muted")
from tensorboardX import SummaryWriter
import torch
import multiprocessing
torch.manual_seed(seed)
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support, \
    classification_report
from sklearn.model_selection import train_test_split
from input_pipeline import imdbTrainDataset, imdbTestDataset
from model import RNN

# TODO: add class weighted validation metrics for testing? should be easy x)
# TODO: roc curves or confusion matrices on the testing data, the latter should be more straightforward
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="path to data")
parser.add_argument("--feats", type=str, help="path to features")
parser.add_argument("--epochs", type=int, help="number of training epochs", default=10)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("--batch_sz", type=int, help="batch size", default=1)
parser.add_argument("--null", type=str, help="path to null features")
parser.add_argument("--ncores", type=int, help="number of cores to use for multiprocessing of training", default=multiprocessing.cpu_count())
parser.add_argument("--oversample", type=str, help="whether to oversample the minority class", default=None)
parser.add_argument("--p", type=float,help="dropout probability", default=0.5)
parser.add_argument("--model", type=str, help="name of model to use for training", default="Net")


def train(model, dataloader, optimizer):
    train_losses = []
    train_precisions = []
    train_f1s = []
    train_recalls = []
    train_accs = []

    start_train_clock = time.clock()
    for batch_number, batch in tqdm.tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        batch_xs = Variable(batch[0].float(), requires_grad=False)
        batch_ys = Variable(batch[1].long(), requires_grad=False)

        # Forward pass: compute output of the network by passing x through the model.
        y_pred_probs = model(batch_xs.long())

        y_pred = np.argmax(y_pred_probs.data.numpy(),axis=1)
        y_test = np.argmax(batch_ys.data.numpy(),axis=0)

        # Compute loss.
        train_loss = loss_fn(y_pred_probs.view(-1), batch_ys.float())

        # Backward pass: compute gradient of the loss with respect to model parameters
        train_loss.backward(retain_graph=True)
        optimizer.step()
    stop_train_clock = time.clock()
    print("epoch time: %d".format(stop_train_clock-start_train_clock))
    # witch the model to evaluation mode (training=False) in order to evaluate on the validation set

if __name__ == '__main__':
    args = parser.parse_args()
    time_stamp = time.time()
    # writer = SummaryWriter("logs/"+str(time_stamp)+"/")

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    # n_bins is the number of class bins
    # num_epochs is the number of complete iterations over the data in batch_size increments

    num_epochs = args.epochs
    batch_size = args.batch_sz
    learning_rate = args.lr
    n_bins = 2
    num_workers = multiprocessing.cpu_count()

    # load the training data, then further partition into training and validation sets, preserving the ratio of
    # positives to negative training examples
    train_data = imdbTrainDataset()
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

    # define the network dimensions based on input data dimensionality
    # N, D_in, H, D_out = batch_size, train_data.data.shape[1], 5, n_bins

    # load the model
    model = RNN(vocab_size=20000, embedding_dim=100, hidden_dim=50,label_size=1, batch_size=2,seq_len=250)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_name = "bce"

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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