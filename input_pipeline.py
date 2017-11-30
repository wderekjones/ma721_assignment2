from torch.utils.data import Dataset
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences


class imdbTrainDataset(Dataset):

    def __init__(self, vocab_size=20000, maxlen=250):
        train, _ = imdb.load_data(num_words=vocab_size, maxlen=maxlen)
        self.data = train[0]
        self.labels = train[1]
        self.data = pad_sequences(self.data, maxlen=250)
        # self.data = self.data.flatten()
        self.labels = self.labels.flatten()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


class imdbTestDataset(Dataset):

    def __init__(self, vocab_size=20000, maxlen=250):
        _, test = imdb.load_data(num_words=vocab_size, maxlen=maxlen)
        self.data = test[0]
        self.labels = test[1]
        self.data = pad_sequences(self.data, maxlen=250)
        # self.data = self.data.flatten()
        self.labels = self.labels.flatten()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


# train_data = imdbTrainDataset()
# test_data = imdbTestDataset()
