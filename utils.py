import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import accuracy_score
from metrics import ye_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int,  default=20000)
    parser.add_argument("--maxLen", type=int, default=250)
    parser.add_argument("--embed", type=int,  default=128)
    parser.add_argument("--hidden", type=int, default=50)
    parser.add_argument("--output", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--keep", type=float, default=0.5)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="debug")
    args = parser.parse_args()
    return args


def output_performance(model, y_pred, y_test):
    num_embedding_weights = (model.layers[0].get_weights()[0].shape[0] * model.layers[-1].get_weights()[0].shape[1])
    acc = accuracy_score(y_true=y_test, y_pred=np.round(y_pred))
    print("testing accuracy: {0:.3f}".format(acc))
    print("ye score: {0:.3f}".format(ye_score(acc, model.count_params())))
    print("adjusted ye score: {0:.3f}".format(ye_score(acc, (model.count_params()-num_embedding_weights))))


def generate_figures(history, model_name, output_dir):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.title("training loss vs. epoch")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(str(output_dir)+"/loss_"+str(model_name)+".png")

    plt.clf()
    plt.plot(history.history['val_loss'])
    plt.title("validation loss vs. epoch")
    plt.ylabel("val loss")
    plt.xlabel("epoch")
    plt.savefig(str(output_dir)+"/val_loss_"+str(model_name)+".png")

    plt.clf()
    plt.plot(history.history['acc'])
    plt.title("training accuracy vs. epoch")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.savefig(str(output_dir)+"/acc_"+str(model_name)+".png")

    plt.clf()
    plt.plot(history.history['val_acc'])
    plt.title("validation accuracy vs. epoch")
    plt.ylabel("val acc")
    plt.xlabel("epoch")
    plt.savefig(str(output_dir)+"/val_acc_" + str(model_name) + ".png")
