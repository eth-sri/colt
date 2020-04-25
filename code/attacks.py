import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import clamp_image
from layers import Conv2d, Flatten, ReLU, Normalization, Linear


def attack_latent(eps, net, bounds, blocks, post_blocks, inputs, targets, n_steps, step_size, detach=True, loss_fn=F.cross_entropy):
    adv_errors = []
    for it in range(n_steps):
        net.zero_grad()
        curr_head, A_0 = clamp_image(inputs, eps)
        if it == 0:
            adv_errors = [torch.FloatTensor(curr_head.size()).uniform_(-1, 1).to(inputs.device).requires_grad_(True)]
        curr_errors = [A_0 * adv_errors[0]]
        err_idx = 0

        for j, layer in enumerate(blocks):
            if isinstance(layer, Conv2d):
                conv = layer.conv
                curr_head = conv(curr_head)
                for i in range(len(curr_errors)):
                    curr_errors[i] = F.conv2d(curr_errors[i], conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
            elif isinstance(layer, ReLU):
                D = 1e-6
                lb, ub = bounds[j]
                is_cross = (lb < 0) & (ub > 0)
                max_relu_lambda = torch.where(is_cross, ub/(ub-lb+D), (lb >= 0).float())

                relu_lambda_cross = max_relu_lambda
                relu_mu_cross = -0.5*ub*lb/(ub-lb)

                relu_lambda = torch.where(is_cross, relu_lambda_cross, (lb >= 0).float())
                relu_mu = torch.where(is_cross, relu_mu_cross, torch.zeros(lb.size()).to(inputs.device))

                curr_head = curr_head * relu_lambda + relu_mu
                for i in range(len(curr_errors)):
                    curr_errors[i] *= relu_lambda
                if it == 0:
                    adv_errors += [torch.FloatTensor(curr_head.size()).uniform_(-1, 1).to(inputs.device).requires_grad_(True)]
                err_idx += 1
                curr_errors += [relu_mu * adv_errors[err_idx]]
            elif isinstance(layer, Normalization):
                curr_head = (curr_head - layer.mean) / layer.sigma
                for i in range(len(curr_errors)):
                    curr_errors[i] /= layer.sigma
            elif isinstance(layer, Flatten):
                curr_head = curr_head.view(curr_head.size()[0], -1)
                for i in range(len(curr_errors)):
                    curr_errors[i] = curr_errors[i].view(curr_errors[i].size()[0], -1)
            elif isinstance(layer, Linear):
                curr_head = layer.linear(curr_head)
                for i in range(len(curr_errors)):
                    curr_errors[i] = torch.matmul(curr_errors[i], layer.linear.weight.t())
            else:
                assert False, 'Unknown layer type!'

        adv_latent = curr_head.clone()
        for i in range(len(curr_errors)):
            adv_latent += curr_errors[i]
        if it == n_steps-1:
            break
        adv_outs = post_blocks(adv_latent)
        ce_loss = loss_fn(adv_outs, targets, reduction='sum')
        ce_loss.backward()
        for i in range(len(adv_errors)):
            adv_errors[i].data = torch.clamp(adv_errors[i].data + step_size * adv_errors[i].grad.sign(), -1, 1)
            adv_errors[i].grad.zero_()
        adv_latent = curr_head.clone()
        for i in range(len(curr_errors)):
            adv_latent += curr_errors[i]
    return adv_latent


def compute_bounds_approx(eps, blocks, layer_idx, inputs, k=50):
    bounds_approx = {}
    batch_size = inputs.size()[0]
    curr_head, A_0 = clamp_image(inputs, eps)
    curr_cauchy = A_0.unsqueeze(0) * torch.clamp(torch.FloatTensor(k, *inputs.size()).to(inputs.device).cauchy_(), -1e10, 1e10)
    n_cauchy = 1
    for j in range(layer_idx+1):
        layer = blocks[j]
        if isinstance(layer, Normalization):
            curr_head = (curr_head - layer.mean) / layer.sigma
            curr_cauchy /= layer.sigma.unsqueeze(0)
        elif isinstance(layer, Conv2d):
            conv = layer.conv
            curr_head = conv(curr_head)
            tmp_cauchy = curr_cauchy.view(-1, *curr_cauchy.size()[2:])
            tmp_cauchy = F.conv2d(tmp_cauchy, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
            curr_cauchy = tmp_cauchy.view(-1, batch_size, *tmp_cauchy.size()[1:])
        elif isinstance(layer, ReLU):
            lb, ub = bounds_approx[j]
            is_cross = (lb < 0) & (ub > 0)
            relu_lambda = torch.where(is_cross, ub/(ub-lb), (lb >= 0).float())
            relu_mu = torch.where(is_cross, -0.5*ub*lb/(ub-lb), torch.zeros(lb.size()).to(inputs.device))
            curr_head = curr_head * relu_lambda + relu_mu
            curr_cauchy = curr_cauchy * relu_lambda.unsqueeze(0)
            new_cauchy = relu_mu.unsqueeze(0) * torch.clamp(torch.FloatTensor(k, *curr_head.size()).to(inputs.device).cauchy_(), -1e10, 1e10)
            curr_cauchy = torch.cat([curr_cauchy, new_cauchy], dim=0)
            n_cauchy += 1
        elif isinstance(layer, Flatten):
            curr_head = curr_head.view(batch_size, -1)
            curr_cauchy = curr_cauchy.view(curr_cauchy.size()[0], batch_size, -1)
        elif isinstance(layer, Linear):
            curr_head = layer.linear(curr_head)
            curr_cauchy = torch.matmul(curr_cauchy, layer.linear.weight.t())
        elif isinstance(layer, nn.BatchNorm2d):
            curr_head = layer(curr_head)
            tmp_cauchy = curr_cauchy.view(-1, *curr_cauchy.size()[2:])
            tmp_cauchy = layer(tmp_cauchy)
            curr_cauchy = tmp_cauchy.view(-1, batch_size, *tmp_cauchy.size()[1:]) - layer.bias.view((1, 1, -1, 1, 1))
        else:
            assert False, 'Unknown layer type!'

        if j+1 < len(blocks) and isinstance(blocks[j+1], ReLU):
            l1_approx = 0
            for i in range(n_cauchy):
                l1_approx += torch.median(curr_cauchy[i*k:(i+1)*k].abs(), dim=0)[0]
            lb = curr_head - l1_approx
            ub = curr_head + l1_approx
            bounds_approx[j+1] = (lb, ub)

    return bounds_approx
