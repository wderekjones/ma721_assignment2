from keras.models import Sequential
from keras.layers import Dense,Conv1D,LSTM, Embedding, Flatten, Dropout, BatchNormalization, MaxPool1D
from keras import regularizers


def cnn(vocab_size, maxLen, embedding_dim, kernel_size, hidden_dim, output_dim, keep_prob):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxLen))
    model.add(BatchNormalization())
    model.add(Dropout(rate=keep_prob))
    model.add(Conv1D(filters=hidden_dim, kernel_size=kernel_size, activation='relu', kernel_regularizer= regularizers.l1(1e-1)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=keep_prob))
    model.add(MaxPool1D())
    model.add(Flatten())
    model.add(Dense(output_dim,activation='relu'))
    model.add(Dense(output_dim, activation='sigmoid', activity_regularizer=regularizers.l1(1e-4)))
    return model


def rnn(vocab_size, maxLen, embedding_dim, hidden_dim, output_dim, batch_size, keep_prob):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxLen))
    model.add(Dropout(rate=keep_prob))
    model.add(LSTM(hidden_dim, batch_size=batch_size, dropout=keep_prob, recurrent_dropout=keep_prob))
    model.add(Dropout(rate=keep_prob))
    model.add(Dense(output_dim, activation='sigmoid'))
    return model
