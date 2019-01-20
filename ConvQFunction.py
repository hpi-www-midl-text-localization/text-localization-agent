import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl


class QFunction(chainer.Chain):

    def __init__(self, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


class ConvQFunction(chainer.ChainList):
    def __init__(self):
        super(ConvQFunction, self).__init__(
            VGG2Block(64),
            VGG2Block(128),
            VGG3Block(256),
            VGG3Block(512),
            VGG3Block(512),
            FCBlock())

    def forward(self, x):
        # x, history = state[0]
        # x = np.array([x])
        # i = 0
        for f in self.children():
            # if i == 4:
            #    x = F.concat((x.reshape(-1), history), 0).reshape((1, -1))
            x = f(x)
            # i += 1

        return chainerrl.action_value.DiscreteActionValue(x)


class VGG2Block(chainer.Chain):
    def __init__(self, n_channels):
        w = chainer.initializers.HeNormal()
        super(VGG2Block, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_channels, 3, 1, 1, initialW=w)
            self.conv2 = L.Convolution2D(
                n_channels, n_channels, 3, 1, 1, initialW=w)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        return h


class VGG3Block(chainer.Chain):
    def __init__(self, n_channels):
        w = chainer.initializers.HeNormal()
        super(VGG3Block, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_channels, 3, 1, 1, initialW=w)
            self.conv2 = L.Convolution2D(
                n_channels, n_channels, 3, 1, 1, initialW=w)
            self.conv3 = L.Convolution2D(
                n_channels, n_channels, 3, 1, 1, initialW=w)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2, 2)
        return h


class FCBlock(chainer.Chain):
    def __init__(self):
        w = chainer.initializers.HeNormal()
        super(FCBlock, self).__init__()
        with self.init_scope():
            self.fc4 = L.Linear(None, 4096, initialW=w)
            self.fc5 = L.Linear(4096, 4096, initialW=w)
            self.fc6 = L.Linear(4096, 9, initialW=w)

    def forward(self, x):
        h = F.dropout(F.relu(self.fc4(x)))
        h = F.dropout(F.relu(self.fc5(h)))
        h = self.fc6(h)
        return h
