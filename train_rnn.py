import numpy as np
np.random.RandomState(0)
from model import rnn
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import optimizers
from utils import output_performance, generate_figures, get_args

args = get_args()

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", num_words=args.vocab_size, maxlen=args.maxLen)

x_train = sequence.pad_sequences(x_train, maxlen=args.maxLen)
x_test = sequence.pad_sequences(x_test, maxlen=args.maxLen)

model = rnn(vocab_size=args.vocab_size, maxLen=args.maxLen, embedding_dim=args.embed,
            hidden_dim=args.hidden, output_dim=args.output, batch_size=args.batch, keep_prob=args.keep)

model.compile(optimizer=optimizers.Adam(lr=args.lr), loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())
history = model.fit(x_train, y_train, validation_split=args.val_split, batch_size=args.batch, epochs=args.epochs)

y_pred = model.predict(x_test)
generate_figures(history=history, model_name="rnn", output_dir="figures")
output_performance(model=model, y_test=y_test, y_pred=y_pred)
