import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        modules = [nn.Conv2d(in_channels, out_channels, k_size, padding=padding, bias=True, stride=stride),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(True)]

        weight_init(modules[0])

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)


class NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, model_depth=4):
        super(NetBlock, self).__init__()
        channels_jump = 2
        modules = []
        cur_channels = in_channels
        next_channels = cur_channels
        for depth in range(model_depth):
            next_channels = cur_channels + channels_jump if \
                (next_channels < (out_channels-channels_jump)) else out_channels
            modules.append(ConvBlock(cur_channels, next_channels))
            cur_channels = next_channels
        modules.append(ConvBlock(cur_channels, out_channels))

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def reset_params(net):
    for i, m in enumerate(net.modules()):
        weight_init(m)


class DeepCT(nn.Module):
    def __init__(self, in_channels, out_channels, model_depth=4):
        super(DeepCT, self).__init__()
        self.net = NetBlock(in_channels=in_channels, out_channels=out_channels, model_depth=model_depth)
        weight_init(self.net)

    def forward(self, x):
        return self.net.forward(x)
