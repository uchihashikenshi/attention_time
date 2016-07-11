# coding:utf-8
from matplotlib import pyplot as plt


class BaseEstimator(object):

    def __init__(self):
        pass

    def param_new(self, rows, cols, output_dim):
        self.rows = rows
        self.cols = cols
        self.output_dim = output_dim

    def data_transform(self, train, test):
        x_train = train['x'].reshape(train['x'].shape[0], self.rows, self.cols)
        x_test = test['x'].reshape(test['x'].shape[0], self.rows, self.cols)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = train['y']
        y_test = test['y']

        return x_train, y_train, x_test, y_test

    #todo: visualize functions are not suit for manny results
    def visualize_acc(self, result, epoch):
        x = range(epoch)
        plt.plot(x, result.history['acc'], label="train acc")
        plt.plot(x, result.history['val_acc'], label="val acc")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def visualize_loss(self, result, epoch):
        x = range(epoch)
        plt.plot(x, result.history['loss'], label="train loss")
        plt.plot(x, result.history['val_loss'], label="val loss")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
