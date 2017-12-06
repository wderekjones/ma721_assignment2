import torch
import torch.nn as nn
from torch import autograd
from torch.nn import Embedding, LSTM
from torch.autograd import Variable
import torch.nn.functional as F

class RNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, seq_len, label_size, batch_size):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_size = label_size
        self.batch_size = batch_size
        self.seq_len = seq_len


        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(input_size=self.seq_len, hidden_size=hidden_dim, num_layers=1)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.fc1 = nn.Linear((self.embedding_dim*self.hidden_dim),self.label_size)
        self.relu = nn.ReLU()
        self.hidden = self.init_hidden()
 
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        x = embeds.view(self.embedding_dim, self.batch_size, self.seq_len*self.batch_size) # reshape output tensor to seq_len, batch_size, input_size for input to lstm module
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        z = lstm_out.view(self.batch_size, -1) # flatten the tensor
        # y = self.hidden2label(lstm_out[-1])
        y = self.fc1(z)
        y = self.relu(y)
        return y


class MLP(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(MLP,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.fc1 = nn.Linear

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        x = embeds.view(1,-1)

