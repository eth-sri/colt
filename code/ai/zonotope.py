"""
Based on HybridZonotope from DfifAI (https://github.com/eth-sri/diffai/blob/master/ai.py)
"""
import numpy as np
import torch
import torch.nn.functional as F


def clamp_image(x, eps):
    min_x = torch.clamp(x-eps, min=0)
    max_x = torch.clamp(x+eps, max=1)
    x_center = 0.5 * (max_x + min_x)
    x_beta = 0.5 * (max_x - min_x)
    return x_center, x_beta


def get_new_errs(should_box, newhead, newbeta):
    new_err_pos = (should_box.long().sum(dim=0) > 0).nonzero()
    num_new_errs = new_err_pos.size()[0]
    nnz = should_box.nonzero()
    if len(newhead.size()) == 2:
        batch_size, n = newhead.size()[0], newhead.size()[1]
        ids_mat = torch.zeros(n, dtype=torch.long).to(newhead.device)
        ids_mat[new_err_pos[:, 0]] = torch.arange(num_new_errs).to(newhead.device)
        beta_values = newbeta[nnz[:, 0], nnz[:, 1]]
        new_errs = torch.zeros((num_new_errs, batch_size, n)).to(newhead.device, dtype=newhead.dtype)
        err_ids = ids_mat[nnz[:, 1]]
        new_errs[err_ids, nnz[:, 0], nnz[:, 1]] = beta_values
    else:
        batch_size, n_channels, img_dim = newhead.size()[0], newhead.size()[1], newhead.size()[2]
        ids_mat = torch.zeros((n_channels, img_dim, img_dim), dtype=torch.long).to(newhead.device)
        ids_mat[new_err_pos[:, 0], new_err_pos[:, 1], new_err_pos[:, 2]] = torch.arange(num_new_errs).to(newhead.device)
        beta_values = newbeta[nnz[:, 0], nnz[:, 1], nnz[:, 2], nnz[:, 3]]
        new_errs = torch.zeros((num_new_errs, batch_size, n_channels, img_dim, img_dim)).to(newhead.device, dtype=newhead.dtype)
        err_ids = ids_mat[nnz[:, 1], nnz[:, 2], nnz[:, 3]]
        new_errs[err_ids, nnz[:, 0], nnz[:, 1], nnz[:, 2], nnz[:, 3]] = beta_values
    return new_errs


class HybridZonotope:

    def __init__(self, head, beta, errors, domain):
        self.head = head
        self.beta = beta
        self.errors = errors
        self.domain = domain
        self.device = self.head.device
        assert not torch.isnan(self.head).any()
        assert self.beta is None or (not torch.isnan(self.beta).any())
        assert self.errors is None or (not torch.isnan(self.errors).any())

    @staticmethod
    def zonotope_from_noise(x, eps, domain, dtype=torch.float32):
        batch_size = x.size()[0]
        n_elements = x[0].numel()
        ei = torch.eye(n_elements).expand(batch_size, n_elements, n_elements).permute(1, 0, 2).to(x.device)
        x_center, x_beta = clamp_image(x, eps)
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        if len(x.size()) > 2:
            ei = ei.contiguous().view(n_elements, *x.size())
        return HybridZonotope(x_center, None, ei * x_beta.unsqueeze(0), domain)

    @staticmethod
    def box_from_noise(x, eps):
        x_center, x_beta = clamp_image(x, eps)
        return HybridZonotope(x_center, x_beta, None, 'zono')

    def size(self):
        return self.head.size()

    def view(self, size):
        return HybridZonotope(self.head.view(*size),
                              None if self.beta is None else self.beta.view(size),
                              None if self.errors is None else self.errors.view(self.errors.size()[0], *size),
                              self.domain)

    def normalize(self, mean, sigma):
        return (self - mean) / sigma

    def __sub__(self, other):
        if isinstance(other, torch.Tensor):
            return HybridZonotope(self.head - other, self.beta, self.errors, self.domain)
        else:
            assert False, 'Unknown type of other object'

    def __add__(self, other):
        if isinstance(other, torch.Tensor):
            return HybridZonotope(self.head + other, self.beta, self.errors, self.domain)
        else:
            assert False, 'Unknown type of other object'

    def __truediv__(self, other):
        if isinstance(other, torch.Tensor):
            return HybridZonotope(self.head / other,
                                  None if self.beta is None else self.beta / abs(other),
                                  None if self.errors is None else self.errors / other,
                                  self.domain)
        else:
            assert False, 'Unknown type of other object'

    def clone(self):
        return HybridZonotope(self.head.clone(),
                              None if self.beta is None else self.beta.clone(),
                              None if self.errors is None else self.errors.clone(),
                              self.domain)

    def detach(self):
        return HybridZonotope(self.head.detach(),
                              None if self.beta is None else self.beta.detach(),
                              None if self.errors is None else self.errors.detach(),
                              self.domain)

    def avg_pool2d(self, kernel_size, stride):
        new_head = F.avg_pool2d(self.head, kernel_size, stride)
        new_beta = None if self.beta is None else F.avg_pool2d(self.beta.view(-1, *self.head.shape[1:]), kernel_size, stride)
        new_errors = None if self.errors is None else F.avg_pool2d(self.errors.view(-1, *self.head.shape[1:]), kernel_size, stride)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def conv2d(self, weight, bias, stride, padding, dilation, groups):
        new_head = F.conv2d(self.head, weight, bias, stride, padding, dilation, groups)
        new_beta = None if self.beta is None else F.conv2d(self.beta, weight.abs(), None, stride, padding, dilation, groups)
        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.errors.size()[2:])
            new_errors = F.conv2d(errors_resized, weight, None, stride, padding, dilation, groups)
            new_errors = new_errors.view(self.errors.size()[0], self.errors.size()[1], *new_errors.size()[1:])
        else:
            new_errors = None
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def linear(self, weight, bias):
        return self.matmul(weight.t()) + bias.unsqueeze(0)

    def matmul(self, other):
        return HybridZonotope(self.head.matmul(other),
                              None if self.beta is None else self.beta.matmul(other.abs()),
                              None if self.errors is None else self.errors.matmul(other),
                              self.domain)

    def relu(self, deepz_lambda, bounds, init_lambda):
        if self.errors is None:
            min_relu, max_relu = F.relu(self.head - self.beta), F.relu(self.head + self.beta)
            return HybridZonotope(0.5 * (max_relu + min_relu), 0.5 * (max_relu - min_relu), None, self.domain)
        assert self.beta is None
        delta = torch.sum(torch.abs(self.errors), 0)
        lb, ub = self.head - delta, self.head + delta

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)
        is_cross = (lb < 0) & (ub > 0)

        D = 1e-6
        relu_lambda = torch.where(is_cross, ub/(ub-lb+D), (lb >= 0).float())
        if self.domain == 'zono_iter':
            if init_lambda:
                # print(relu_lambda.size())
                # print(deepz_lambda.size())
                deepz_lambda.data = relu_lambda.data.squeeze(0)

            assert (deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()

            relu_lambda_cross = deepz_lambda.unsqueeze(0)
            relu_mu_cross = torch.where(relu_lambda_cross < relu_lambda, 0.5*ub*(1-relu_lambda_cross), -0.5*relu_lambda_cross*lb)

            # relu_lambda_cross = deepz_lambda * relu_lambda
            # relu_mu_cross = 0.5*ub*(1-relu_lambda_cross)

            # relu_lambda_cross = relu_lambda + (1 - deepz_lambda) * (1 - relu_lambda)
            # relu_mu_cross = -0.5*relu_lambda_cross*lb

            relu_lambda = torch.where(is_cross, relu_lambda_cross, (lb >= 0).float())
            relu_mu = torch.where(is_cross, relu_mu_cross, torch.zeros(lb.size()).to(self.device))
        else:
            relu_mu = torch.where(is_cross, -0.5*ub*lb/(ub-lb+D), torch.zeros(lb.size()).to(self.device))

        assert (not torch.isnan(relu_mu).any()) and (not torch.isnan(relu_lambda).any())

        new_head = self.head * relu_lambda + relu_mu
        old_errs = self.errors * relu_lambda
        new_errs = get_new_errs(is_cross, new_head, relu_mu)
        new_errors = torch.cat([old_errs, new_errs], dim=0)
        assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
        return HybridZonotope(new_head, None, new_errors, self.domain)

    def concretize(self):
        delta = 0
        if self.beta is not None:
            delta = delta + self.beta
        if self.errors is not None:
            delta = delta + self.errors.abs().sum(0)
        return self.head - delta, self.head + delta

    def avg_width(self):
        lb, ub = self.concretize()
        return (ub - lb).mean()

    def is_greater(self, i, j):
        if self.errors is not None:
            diff_errors = (self.errors[:, :, i] - self.errors[:, :, j]).abs().sum(dim=0)
            diff_head = self.head[:, i] - self.head[:, j]
            delta = diff_head - diff_errors
            if self.beta is not None:
                delta -= self.beta[:, i].abs() + self.beta[:, j].abs()
            return delta, delta > 0
        else:
            diff_head = (self.head[:, i] - self.head[:, j])
            diff_beta = (self.beta[:, i] + self.beta[:, j]).abs()
            delta = (diff_head - diff_beta)
            return delta, delta > 0

    def verify(self, targets):
        n_class = self.head.size()[1]
        verified = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        verified_corr = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        for i in range(n_class):
            isg = torch.ones(targets.size(), dtype=torch.uint8).to(self.head.device)
            for j in range(n_class):
                if i != j:
                    _, ok = self.is_greater(i, j)
                    isg = isg & ok.byte()
            verified = verified | isg
            verified_corr = verified_corr | (targets.eq(i).byte() & isg)
        return verified, verified_corr

    def get_min_diff(self, i, j):
        """ returns minimum of logit[i] - logit[j] """
        return self.is_greater(i, j)[0]

    def get_wc_logits(self, targets):
        batch_size = targets.size()[0]
        lb, ub = self.concretize()
        wc_logits = ub
        wc_logits[np.arange(batch_size), targets] = lb[np.arange(batch_size), targets]
        return wc_logits

    def ce_loss(self, targets):
        wc_logits = self.get_wc_logits(targets)
        return F.cross_entropy(wc_logits, targets)
