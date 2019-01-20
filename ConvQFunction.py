import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl


class ConvQFunction(chainer.ChainList):
    def __init__(self):
        super(ConvQFunction, self).__init__(
            VGGBlock(64),
            VGGBlock(128),
            VGGBlock(256, 3),
            VGGBlock(512, 3),
            VGGBlock(512, 3),
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


class VGGBlock(chainer.Chain):
    def __init__(self, n_channels, n_convs=2):
        w = chainer.initializers.HeNormal()
        super(VGGBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, n_channels, 3, 1, 1, initialW=w)
            self.conv2 = L.Convolution2D(
                n_channels, n_channels, 3, 1, 1, initialW=w)
            if n_convs == 3:
                self.conv3 = L.Convolution2D(
                    n_channels, n_channels, 3, 1, 1, initialW=w)

        self.n_convs = n_convs

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        if self.n_convs == 3:
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
