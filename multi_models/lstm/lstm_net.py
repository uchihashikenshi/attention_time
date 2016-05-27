# coding:utf-8

class LSTM(chainer.Chain):
    def __init__(self, inp_size, n_units, train=True):
        super(LSTM, self).__init__(
            embed = L.EmbedID(inp_size, n_units),]
            l1 = L.LSTM(inp_size, n_units),
            l2 = L.Linear(n_units, inp_size),
        )

    def __call__(self):
        h0 = self.embed(x)
        h1 = self.l1(h0)
        y = self.l2(h1)

        return y

    def reset_status(self):
        self.l1.reset_state()