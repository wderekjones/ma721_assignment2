import torch.nn as nn
from torch.nn import Embedding, LSTM


class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, 2)
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.fc1(x)
        x = self.sigmoid1(x)

        return x

# model = RNN(250,100,20)
