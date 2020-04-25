import torch.nn as nn
import torch
from loaders import get_mean_sigma
from layers import Conv2d, Normalization, ReLU, Flatten, Linear, Sequential


class SeqNet(nn.Module):

    def __init__(self):
        super(SeqNet, self).__init__()
        self.is_double = False
        self.skip_norm = False

    def forward(self, x, init_lambda=False):
        if isinstance(x, torch.Tensor) and self.is_double:
            x = x.to(dtype=torch.float64)
        x = self.blocks(x, init_lambda, skip_norm=self.skip_norm)
        return x

    def reset_bounds(self):
        for block in self.blocks:
            block.bounds = None

    def to_double(self):
        self.is_double = True
        for param_name, param_value in self.named_parameters():
            param_value.data = param_value.data.to(dtype=torch.float64)

    def forward_until(self, i, x):
        """ Forward until layer i (inclusive) """
        x = self.blocks.forward_until(i, x)
        return x

    def forward_from(self, i, x):
        """ Forward from layer i (exclusive) """
        x = self.blocks.forward_from(i, x)
        return x


class FFNN(SeqNet):

    def __init__(self, device, dataset, sizes, n_class=10, input_size=32, input_channel=3):
        super(FFNN, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)
        self.normalizer = Normalization(mean, sigma)

        layers = [Flatten(), Linear(input_size*input_size*input_channel, sizes[0]), ReLU(sizes[0])]
        for i in range(1, len(sizes)):
            layers += [
                Linear(sizes[i-1], sizes[i]),
                ReLU(sizes[i]),
            ]
        layers += [Linear(sizes[-1], n_class)]
        self.blocks = Sequential(*layers)


class ConvMed(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1, linear_size=100):
        super(ConvMed, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16*width1, 5, stride=2, padding=2, dim=input_size),
            ReLU((16*width1, input_size//2, input_size//2)),
            Conv2d(16*width1, 32*width2, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((32*width2, input_size//4, input_size//4)),
            Flatten(),
            Linear(32*width2*(input_size // 4)*(input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)

        
class ConvMedBig(SeqNet):

    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1, width3=1, linear_size=100):
        super(ConvMedBig, self).__init__()

        mean, sigma = get_mean_sigma(device, dataset)
        self.normalizer = Normalization(mean, sigma)

        layers = [
            Normalization(mean, sigma),
            Conv2d(input_channel, 16*width1, 3, stride=1, padding=1, dim=input_size),
            ReLU((16*width1, input_size, input_size)),
            Conv2d(16*width1, 16*width2, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((16*width2, input_size//2, input_size//2)),
            Conv2d(16*width2, 32*width3, 4, stride=2, padding=1, dim=input_size//2),
            ReLU((32*width3, input_size//4, input_size//4)),
            Flatten(),
            Linear(32*width3*(input_size // 4)*(input_size // 4), linear_size),
            ReLU(linear_size),
            Linear(linear_size, n_class),
        ]
        self.blocks = Sequential(*layers)


