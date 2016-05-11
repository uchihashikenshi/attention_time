# coding:utf-8

import chainer
import chainer.functions as F
import chainer.links as L


class tsCNN_net(chainer.Chain):

    def __init__(self, data_dim, output_dim):
        self.data_dim = data_dim
        self.output_dim = output_dim
        super(tsCNN_net, self).__init__(
            conv1=F.Convolution2D(1, 20, (1, 5)),
            conv2=F.Convolution2D(20, 50, (1, 5)),
            l1=L.Linear(250, 500),
            l2=L.Linear(500, self.output_dim)
            )

    def __call__(self, x, t):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2)
        h3 = F.relu(self.l1(h2))
        y = self.l2(h3)

        self.loss = F.sigmoid_cross_entropy(y, t)

        accuracy = []
        y_ = F.array.split_axis.split_axis(y, self.output_dim, 1)
        t_ = F.array.split_axis.split_axis(t, self.output_dim, 1)

        for y_ele, t_ele in zip(y_, t_):
            accuracy.append(F.accuracy(y_ele, chainer.Variable(t_ele.data.squeeze())))
        self.accuracy = accuracy

        return self.loss
