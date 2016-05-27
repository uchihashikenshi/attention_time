# coding:utf-8
import argparse
import math
import sys
import time
import numpy
import six
import chainer
from chainer import optimizers
from chainer import serializers
import chainer.functions as F
import chainer.links as L


class LSTM(chainer.Chain):
    def __init_(self, p, n_units, train=True):
        super(LSTM, self).__init__(
            embed=L.EmbedID(p+1, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.Linear(n_units, p+1),
        )

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        y = self.l2(h1)
        return y

    def reset_state(self):
        self.l1.reset_state()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# parameter
p = 5  # length of sentence
n_units = 4  # unit number of hidden layer

train_data = numpy.ndarray((2, p+1), dtype=numpy.int32)
train_data[0][0] = train_data[0][p] = p
train_data[1][0] = train_data[1][p] = p - 1

for i in xrange(p-1):
    train_data[0][i+1] = i
    train_data[1][i+1] = i

print train_data[0]
print train_data[1]

lstm = LSTM(p, n_units)
model = L.Classifier(lstm)
model.compute_accuracy = False
for param in model.params():
    data = param.data
    data[:] = numpy.random.uniform(-0.2, 0.2, data.shape)

# optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# train
display = 1000
total_loss = 0
for seq in xrange(100000):
    sequence = train_data[numpy.random.randint(2)]
    lstm.reset_state()
    for i in six.moves.range(p):
        x = chainer.Variable(numpy.asarray([sequence[i]]))
        t = chainer.Variable(numpy.asarray([sequence[i+1]]))
        loss = model(x, t)

        if seq % display == 0:
            total_loss += loss.data

        model.zerograds()
        loss.backward()
        optimizer.update()

    if seq % display == 0:
        print "sequence:{}, loss:{}".format(seq, total_loss)
        total_loss = 0

    if seq % (display * 10) == 0:
        for select in six.moves.range(2):
            sequence = train_data[select]
            lstm.reset_state()
            print "prediction:{},".format(sequence[0])
            for i in six.moves.range(p):
                x = chainer.Variable(numpy.asarray([sequence[i]]))
                data = lstm(x).data
                print "{},".format(numpyargmax(data))
            print ""
            print "probability:{}".format(data)