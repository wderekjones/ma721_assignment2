import torch
import torch.nn as nn
from torch import autograd
from torch.nn import LSTM
from torch.autograd import Variable
import torch.nn.functional as F


class smallRNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, seq_len, label_size, batch_size):
        super(smallRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_size = label_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = LSTM(input_size=self.embedding_dim, hidden_size=label_size, num_layers=1, batch_first=True)
        # self.softmax = nn.Softmax()


    def forward(self, inputs):
        h0 = autograd.Variable(torch.zeros(1, self.batch_size, self.label_size)).cuda()
        c0 = autograd.Variable(torch.zeros(1, self.batch_size, self.label_size)).cuda()
        embeds = self.embeddings(inputs)
        # embeds = embeds.mean(dim=1)
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, self.hidden = self.lstm(embeds, (h0,c0))
        y = lstm_out.mean(dim=1)
        # y = self.hidden2label(lstm_out[-1])
        # y = self.fc1(z)
        # y = self.softmax(y)
        return y


class CNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, seq_len, label_size, batch_size):
        super(CNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_size = label_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim,hidden_dim,3)
        self.fc1 = nn.Linear(hidden_dim,label_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        # embeds_reduced = embeds.mean(dim=1)
        x = self.conv1(embeds)
        x = self.fc1(x)
        y = self.sigmoid(x)
        return y

