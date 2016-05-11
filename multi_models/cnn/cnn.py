# coding:utf-8

from __future__ import print_function
import argparse
import time

import numpy
import six

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

# from chainer.functions.loss import sigmoid_cross_entropy

import cnn_net


class CNN(object):

    def __init__(self, save_model_dir, epoch_num=20, batchsize=50):
        self.save_model_dir = save_model_dir
        self.epoch_num = epoch_num
        self.batchsize = batchsize

    def train(self, train, test, output_dim):

        train_x, train_y = train["x"].astype(numpy.float32), train["y"].astype(numpy.int32)
        test_x, test_y = test["x"].astype(numpy.float32), test["y"].astype(numpy.int32)

        N_train = train_y.shape[0]
        N_test = test_y.shape[0]
        data_dim = train_x[0].shape[0]

        train_x = train_x.reshape(N_train, 1, 1, data_dim)
        test_x = test_x.reshape(N_test, 1, 1, data_dim)

        # Prepare multi-layer perceptron model, defined in cnn_net.py
        model = cnn_net.tsCNN_net(data_dim, output_dim)

        # Setup optimizer
        optimizer = optimizers.Adam()
        optimizer.setup(model)

        acc_txt = open(self.save_model_dir + "accuracy.txt", "w")
        loss_txt = open(self.save_model_dir + "loss.txt", "w")

        acc_txt.write("epoch\test_accuracy\n")
        loss_txt.write("epoch\train_loss\n")

        # Learning loop
        for epoch in six.moves.range(1, self.epoch_num + 1):
            print('epoch', epoch)

            # training
            perm = numpy.random.permutation(N_train)
            sum_accuracy = [0 for i in xrange(output_dim)]
            sum_loss = 0
            start = time.time()
            for i in six.moves.range(0, N_train, self.batchsize):
                x = chainer.Variable(numpy.asarray(train_x[perm[i:i + self.batchsize]]))
                t = chainer.Variable(numpy.asarray(train_y[perm[i:i + self.batchsize]]))

                # Pass the loss function (Classifier defines it) and its arguments
                optimizer.update(model, x, t)

                if epoch == 1 and i == 0:
                    with open(self.save_model_dir + 'graph.dot', 'w') as o:
                        g = computational_graph.build_computational_graph(
                            (model.loss, ))
                        o.write(g.dump())
                    print('graph generated')

                sum_loss += float(model.loss.data) * len(t.data)
                for label_ind, accuracy_label in enumerate(model.accuracy):
                    sum_accuracy[label_ind] += float(accuracy_label.data) * len(t.data)
            end = time.time()
            elapsed_time = end - start
            throughput = N_train / elapsed_time
            print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
                sum_loss / N_train, numpy.array(sum_accuracy).astype(numpy.float32) / N_train, throughput))
            loss_txt.write("%d\t%f\n" % (epoch, sum_loss / N_train))
            loss_txt.flush()

            # evaluation
            sum_accuracy = [0 for i in xrange(output_dim)]
            sum_loss = 0
            for i in six.moves.range(0, N_test, self.batchsize):
                # volatile='on' is not needed?
                x = chainer.Variable(numpy.asarray(test_x[i:i + self.batchsize]))
                t = chainer.Variable(numpy.asarray(test_y[i:i + self.batchsize]))
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                for label_ind, accuracy_label in enumerate(model.accuracy):
                    sum_accuracy[label_ind] += float(accuracy_label.data) * len(t.data)

            print('test  mean loss={}, accuracy={}'.format(
                sum_loss / N_test, numpy.array(sum_accuracy).astype(numpy.float32) / N_test))

            acc_txt.write("%d\t" % epoch)
            for i, acc in enumerate(numpy.array(sum_accuracy).astype(numpy.float32) / N_test):
                if i == output_dim - 1:
                    acc_txt.write("%f\n" % acc)
                else:
                    acc_txt.write("%f\t" % acc)
            acc_txt.flush()

        # Save the model and the optimizer
        print('save the model')
        serializers.save_npz(self.save_model_dir + 'mlp.model', model)
        print('save the optimizer')
        serializers.save_npz(self.save_model_dir + 'mlp.state', optimizer)