import numpy as np
np.random.RandomState(0)
from model import cnn
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import optimizers
from sklearn.metrics import accuracy_score

vocab_size = 20000
maxLen = 250
embedding_dim = 128
hidden_dim = 50
output_dim = 1
batch_size = 1000
num_epochs = 100

(x_train, y_train),(x_test, y_test) =imdb.load_data(path="imdb.npz", num_words=vocab_size, maxlen=maxLen)

x_train = sequence.pad_sequences(x_train, maxlen=maxLen)
x_test = sequence.pad_sequences(x_test, maxlen=maxLen)

model = cnn(input_dim=x_train.shape[1], vocab_size=vocab_size, maxLen=maxLen, embedding_dim=embedding_dim,
            hidden_dim=hidden_dim, ouput_dim=output_dim)

model.compile(optimizer=optimizers.Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
# output the model description
print(model.summary())
model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=num_epochs)

y_pred = model.predict(x_test)

acc = accuracy_score(y_true=y_test, y_pred=np.round(y_pred))

print("testing accuracy: {0:.3f}".format(acc))

