#coding:utf-8

import chainer
import chainer.functions as F
import chainer.links as L


class tsCNN_net(chainer.Chain):

    def __init__(self, data_dim):
        self.data_dim = data_dim
        super(tsCNN_net, self).__init__(
            conv1=F.Convolution2D(1, 20, (1, 5)),
            conv2=F.Convolution2D(20, 50, (1, 5)),
            l1=L.Linear(250, 500),
            l2=L.Linear(500, 2)
            )

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2)
        h3 = F.relu(self.l1(h2))
        y = self.l2(h3)
        return y