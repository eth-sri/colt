import torch
import torch.nn as nn
from ai.zonotope import HybridZonotope
from layers import Linear
from networks import ConvMed, FFNN, ConvMedBig


def clamp_image(x, eps):
    min_x = torch.clamp(x-eps, min=0)
    max_x = torch.clamp(x+eps, max=1)
    x_center = 0.5 * (max_x + min_x)
    x_betas = 0.5 * (max_x - min_x)
    return x_center, x_betas


def get_network(device, args, input_size, input_channel, n_class):
    if args.net.startswith('ffnn_'):
        tokens = args.net.split('_')
        sizes = [int(x) for x in tokens[1:]]
        net = FFNN(device, args.dataset, sizes, n_class, input_size, input_channel)
    elif args.net.startswith('convmed_'):
        tokens = args.net.split('_')
        obj = ConvMedBatchNorm if 'batchnorm' in tokens else ConvMed
        width1 = int(tokens[2])
        width2 = int(tokens[3])
        linear_size = int(tokens[4])
        net = obj(device, args.dataset, n_class, input_size, input_channel, width1=width1, width2=width2, linear_size=linear_size)
    elif args.net.startswith('convmedbig_'):
        tokens = args.net.split('_')
        assert tokens[0] == 'convmedbig'
        width1 = int(tokens[2])
        width2 = int(tokens[3])
        width3 = int(tokens[4])
        linear_size = int(tokens[5])
        net = ConvMedBig(device, args.dataset, n_class, input_size, input_channel, width1, width2, width3, linear_size=linear_size)
    else:
        assert False, 'Unknown network!'
    net = net.to(device)
    if args.load_model is not None:
        net.load_state_dict(torch.load(args.load_model))
    return net


def get_inputs(train_domain, inputs, eps, device, n_errors=None, dtype=torch.float32):
    if train_domain == 'box':
        return HybridZonotope.box_from_noise(inputs, eps)
    elif train_domain == 'zono' or train_domain == 'zono_iter':
        return HybridZonotope.zonotope_from_noise(inputs, eps, train_domain)
    else:
        assert False


def compute_l1(net, l1_reg):
    ret = 0
    for layer in net.blocks:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, Linear):
            for param_name, param_value in layer.named_parameters():
                if 'bias' in param_name:
                    continue
                ret += param_value.abs().sum()
    return l1_reg * ret


class Scheduler:

    def __init__(self, start, end, n_steps, warmup):
        self.start = start
        self.end = end
        self.n_steps = n_steps
        self.warmup = warmup
        self.curr_steps = 0

    def advance_time(self, k_steps):
        self.curr_steps += k_steps

    def get(self):
        if self.n_steps == self.warmup:
            return self.end
        if self.curr_steps < self.warmup:
            return self.start
        elif self.curr_steps > self.n_steps:
            return self.end
        return self.start + (self.end - self.start) * (self.curr_steps - self.warmup) / float(self.n_steps - self.warmup)


class Statistics:

    def __init__(self):
        self.n = 0
        self.avg = 0.0

    def update(self, x):
        self.avg = self.avg * self.n / float(self.n + 1) + x / float(self.n + 1)
        self.n += 1

    @staticmethod
    def get_statistics(k):
        return [Statistics() for _ in range(k)]
