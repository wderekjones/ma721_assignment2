from keras.models import Sequential
from keras.layers import Dense,Conv1D,LSTM, Embedding, Flatten, Dropout, BatchNormalization
from keras.regularizers import l1,l2


def baseline(input_dim, vocab_size,maxLen, embedding_dim, hidden_dim, ouput_dim):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxLen))
    model.add(Flatten())
    model.add(Dense(ouput_dim,activation='sigmoid'))
    return model


def cnn(input_dim, vocab_size,maxLen, embedding_dim, hidden_dim, ouput_dim, keep_prob=0.4):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxLen))
    model.add(BatchNormalization())
    model.add(Dropout(rate=keep_prob))
    model.add(Conv1D(filters=ouput_dim, kernel_size=embedding_dim, activation='relu'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(rate=keep_prob))
    model.add(Dense(ouput_dim, activation='sigmoid', activity_regularizer=l1(1e-4)))
    return model


def rnn(input_dim, vocab_size,maxLen, embedding_dim, hidden_dim, ouput_dim, batch_size, keep_prob=0.4):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxLen))
    model.add(Dropout(rate=keep_prob))
    model.add(LSTM(hidden_dim, batch_size=batch_size, dropout=keep_prob, recurrent_dropout=keep_prob))
    model.add(Dropout(rate=keep_prob))
    model.add(Dense(ouput_dim, activation='sigmoid'))
    return model
