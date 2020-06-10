import torch
import torch.nn as nn
import torch.nn.functional as F

class BOWNN(nn.Module):
    def __init__(self, n_components, vocab_size, internal_dim = 64):
        super().__init__()

        self.e = nn.EmbeddingBag(vocab_size, internal_dim, mode = 'max')
        self.out = nn.Linear(internal_dim, n_components, bias = False)

    def forward(self, x):
        x = x.squeeze(1).long()
        x = self.e(x)
        x = self.out(x)

        return x

class SimpleCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # internal_dim = int((in_channels*out_channels)**.5) # geometric mean
        internal_dim = 2*max(in_channels, out_channels)

        self.conv1 = nn.Conv2d(in_channels, internal_dim, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(internal_dim)
        self.conv2 = nn.Conv2d(internal_dim, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.conv2(x)

        return x

class SEBlock(nn.Module):
    def __init__(self, cnn_block, in_channels, out_channels, ratio = 16):
        super().__init__()

        internal_dim = out_channels // ratio

        self.cnn_block = cnn_block

        self.in_pool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(in_channels, internal_dim)
        self.out = nn.Linear(internal_dim, out_channels)

    def forward(self, x):
        layers = self.cnn_block(x)
        se = self.in_pool(x).view(x.shape[0], -1)
        se = torch.relu(self.lin1(se))
        se = torch.sigmoid(self.out(se))
        x = layers * se.view(x.shape[0], -1, 1, 1)

        return x

class CNN(nn.Module):
    def __init__(self, n_components, n_layers, internal_dim = 64, n_channels = 1, p = 0):
        super().__init__()

        self.n_layers = n_layers

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(n_layers):
            in_dim = n_channels if i == 0 else internal_dim*2**(i-1)
            out_dim = internal_dim*2**i
            this_conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1) # simplest version
            # this_conv = SEBlock(nn.Conv2d(in_dim, out_dim, 3, 1, 1), in_dim, out_dim)
            # this_conv = SEBlock(SimpleCNNBlock(in_dim, out_dim), in_dim, out_dim)
            self.conv_layers.append(this_conv)

            if i != n_layers - 1:
                this_pool = nn.MaxPool2d(2)
            else:
                this_pool = nn.AdaptiveAvgPool2d(1)
            self.pool_layers.append(this_pool)

            this_bn = nn.BatchNorm2d(out_dim)
            self.bn_layers.append(this_bn)

        dense_dim = internal_dim*2**(n_layers - 1)
        self.do1 = nn.Dropout(p)
        self.lin1 = nn.Linear(dense_dim, dense_dim)
        self.do2 = nn.Dropout(p)
        self.out = nn.Linear(dense_dim, n_components)

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.bn_layers[i](torch.relu(self.pool_layers[i](self.conv_layers[i](x))))

        x = x.view(x.shape[0], -1)
        x = self.do1(x)
        x = torch.relu(self.lin1(x))
        x = self.do2(x)
        x = self.out(x)

        return x

class FFNN(nn.Module):
    def __init__(self, n_components, input_dim, internal_dim, n_hidden_layers = 2):
        super().__init__()

        self.internal_dim = internal_dim

        self.lin_in = nn.Linear(input_dim, self.internal_dim)

        self.layers = nn.ModuleList()
        for i in range(n_hidden_layers - 1):
            self.layers.append(nn.Linear(self.internal_dim, self.internal_dim))

        self.out = nn.Linear(self.internal_dim, n_components)

    def forward(self, x):
        x  = x.view(x.shape[0], -1)
        x = torch.relu(self.lin_in(x))

        for layer in self.layers:
            x = torch.relu(layer(x))

        x = self.out(x)

        return x

class ClusterNet(nn.Module):
    def __init__(self, n_components, n_classes, scale = 4, p = .3):
        super().__init__()

        self.n_classes = n_classes
        internal_dim = n_classes*scale

        self.lin1 = nn.Linear(n_components, internal_dim)
        self.do1 = nn.AlphaDropout(p)
        self.lin2 = nn.Linear(internal_dim, internal_dim)
        self.do2 = nn.AlphaDropout(p)
        self.lin3 = nn.Linear(internal_dim, internal_dim)
        self.do3 = nn.AlphaDropout(p)
        self.out = nn.Linear(internal_dim, n_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.do1(F.selu(self.lin1(x)))
        x = self.do2(F.selu(self.lin2(x)))
        x = self.do3(F.selu(self.lin3(x)))
        x = self.out(x)

        return x

class FullNet(nn.Module):
    def __init__(self, embed_net, cluster_net):
        super().__init__()

        self.embed_net = embed_net
        self.cluster_net = cluster_net

    def forward(self, x):
        x = self.embed_net(x)
        x = self.cluster_net(x)

        return x