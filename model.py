import torch
import torch.nn as nn
from torch.nn import Embedding, LSTM
from torch import autograd



class ConvNet(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim,label_size):
        super(ConvNet, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_size = label_size
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, label_size)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc1(x.view(-1,self.embedding_dim))
        x = self.sigmoid1(x)
        return x
'''
class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim,label_size):
        super(LSTMClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, label_size)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.fc1(x)
        x = self.sigmoid1(x)

        return x

# model = RNN(250,100,20)
'''
'''
class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        # log_probs = F.log_softmax(y)
        log_probs = nn.LogSoftmax(y)
        return log_probs
'''