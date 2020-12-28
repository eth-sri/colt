import time
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from args_factory import get_args
from loaders import get_loaders
from utils import clamp_image, get_network, get_inputs, Scheduler
from layers import Conv2d, Linear, ReLU, Flatten, Normalization
from tqdm import tqdm
from attacks import compute_bounds_approx

torch.cuda.manual_seed(100)
torch.manual_seed(100)
torch.set_printoptions(precision=10)
np.random.seed(100)


def attack_layer(device, eps, layer_idx, net, bounds, inputs, targets, n_steps, step_size, detach=True, loss_fn=F.cross_entropy):
    adv_errors = []
    for it in range(n_steps):
        net.zero_grad()
        curr_head, A_0 = clamp_image(inputs, eps)
        if it == 0:
            adv_errors = [torch.FloatTensor(curr_head.size()).uniform_(-1, 1).to(device).requires_grad_(True)]
        curr_errors = [A_0 * adv_errors[0]]
        err_idx = 0

        for j, layer in enumerate(net.blocks[:layer_idx+1]):
            if isinstance(layer, Conv2d):
                conv = layer.conv
                curr_head = conv(curr_head)
                for i in range(len(curr_errors)):
                    curr_errors[i] = F.conv2d(curr_errors[i], conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
            elif isinstance(layer, ReLU):
                D = 1e-6
                lb, ub = bounds[j]
                is_cross = (lb < 0) & (ub > 0)
                relu_lambda_cross = torch.where(is_cross, ub/(ub-lb+D), (lb >= 0).float())
                relu_mu_cross = -0.5*ub*lb/(ub-lb)
                relu_lambda = torch.where(is_cross, relu_lambda_cross, (lb >= 0).float())
                relu_mu = torch.where(is_cross, relu_mu_cross, torch.zeros(lb.size()).to(device))
                curr_head = curr_head * relu_lambda + relu_mu
                for i in range(len(curr_errors)):
                    curr_errors[i] *= relu_lambda
                if it == 0:
                    adv_errors += [torch.FloatTensor(curr_head.size()).uniform_(-1, 1).to(device).requires_grad_(True)]
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
            elif isinstance(layer, nn.BatchNorm2d):
                curr_head = layer(curr_head)
                for i in range(len(curr_errors)):
                    curr_errors[i] = layer(curr_errors[i]) - layer.bias.view((1, -1, 1, 1))
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
        adv_outs = net.forward_from(layer_idx, adv_latent)
        ce_loss = loss_fn(adv_outs, targets, reduction='sum')
        ce_loss.backward()
        for i in range(len(adv_errors)):
            adv_errors[i].data = torch.clamp(adv_errors[i].data + step_size * adv_errors[i].grad.sign(), -1, 1)
            adv_errors[i].grad.zero_()
        adv_latent = curr_head.clone()
        for i in range(len(curr_errors)):
            adv_latent += curr_errors[i]
    return adv_latent


def test(device, epoch, args, net, test_loader, layers):
    net.eval()
    test_nat_loss, test_nat_ok, test_pgd_loss, test_pgd_ok, n_batches = 0, 0, {}, {}, 0
    test_abs_width, test_abs_ok, test_abs_loss, test_abs_n, test_abs_ex = {}, {}, {}, {}, {}
    for domain in args.test_domains:
        test_abs_width[domain], test_abs_ok[domain], test_abs_loss[domain], test_abs_n[domain], test_abs_ex[domain] = 0, 0, 0, 0, 0
    for layer_idx in layers:
        test_pgd_loss[layer_idx], test_pgd_ok[layer_idx] = 0, 0
    pbar = tqdm(test_loader)

    relu_params = []
    for param_name, param_value in net.named_parameters():
        if 'deepz_lambda' in param_name:
            relu_params.append(param_value)
            param_value.requires_grad_(True)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        for param in relu_params:
            param.data = 1.0*torch.ones(param.size()).to(device)

        bounds = compute_bounds_approx(args.test_eps, net.blocks, layers[-1], inputs, args.n_rand_proj)

        pgd_loss, pgd_ok = {}, {}
        for layer_idx in layers:
            with torch.enable_grad():
                pgd_loss[layer_idx], pgd_ok[layer_idx] = get_adv_loss(
                    device, args.test_eps, layer_idx, net, bounds, inputs, targets, args.test_att_n_steps, args.test_att_step_size, avg=False)
                test_pgd_loss[layer_idx] += pgd_loss[layer_idx].item()
                test_pgd_ok[layer_idx] += pgd_ok[layer_idx].mean().item()

        for domain in args.test_domains:
            abs_inputs = get_inputs('zono' if domain == 'zono_iter' else domain, inputs, args.test_eps, device)
            abs_out = net(abs_inputs)
            abs_loss = abs_out.ce_loss(targets)
            abs_width = abs_out.avg_width().item()
            verified, verified_corr = abs_out.verify(targets)
            test_abs_loss[domain] += abs_loss.item()
            test_abs_width[domain] += abs_width
            test_abs_ok[domain] += verified_corr.float().mean().item()
            test_abs_n[domain] += 1
            for layer_idx in layers:
                # print(verified_corr, pgd_ok[layer_idx])
                assert (verified_corr <= pgd_ok[layer_idx]).all()
        nat_outs = net(inputs)
        nat_loss = F.cross_entropy(nat_outs, targets)
        test_nat_loss += nat_loss.item()
        test_nat_ok += targets.eq(nat_outs.max(dim=1)[1]).float().mean().item()
        n_batches += 1
        abs_ok_str = ', '.join(['%s: %.4f' % (domain, test_abs_ok[domain]/n_batches) for domain in args.test_domains])
        abs_width_str = ', '.join(['%s: %.4f' % (
            domain, -1 if test_abs_n[domain] == 0 else test_abs_width[domain]/test_abs_n[domain]) for domain in args.test_domains])
        abs_pgd_ok_str = ', '.join(['%d: %.4f' % (layer_idx, test_pgd_ok[layer_idx]/n_batches) for layer_idx in layers])
        abs_pgd_loss_str = ', '.join(['%d: %.4f' % (layer_idx, test_pgd_loss[layer_idx]/n_batches) for layer_idx in layers])
        pbar.set_description('[V] nat_loss=%.4f, nat_ok=%.4f, pgd_loss={%s}, pgd_ok={%s}' % (
            test_nat_loss/n_batches,
            test_nat_ok/n_batches,
            abs_pgd_loss_str, abs_pgd_ok_str))
    return test_nat_loss/n_batches, test_nat_ok/n_batches, test_pgd_loss[layers[0]]/n_batches, test_pgd_ok[layers[0]]/n_batches


def compute_bounds(net, device, layer_idx, args, abs_inputs):
    bounds = {}
    abs_curr = abs_inputs
    for j, layer in enumerate(net.blocks[:layer_idx+1]):
        lb, ub = abs_curr.concretize()
        bounds[j] = (lb, ub)
        abs_curr = layer(abs_curr)
    return bounds


def compute_l1_loss(net):
    l1_loss = 0
    for param_name, param_value in net.named_parameters():
        if 'weight' in param_name:
            l1_loss += param_value.abs().sum()
    return l1_loss


def get_adv_loss(device, eps, layer_idx, net, bounds, inputs, targets, n_steps, step_size, detach=True, loss_fn=F.cross_entropy, avg=True, is_train=False):
    adv_latent = attack_layer(device, eps, layer_idx, net, bounds, inputs, targets, n_steps, step_size, detach, loss_fn)
    if detach:
        adv_latent = adv_latent.clone().detach()
    net.zero_grad()
    if is_train:
        net.train()
    adv_outs = net.forward_from(layer_idx, adv_latent)
    adv_loss = loss_fn(adv_outs, targets)
    adv_ok = targets.eq(adv_outs.max(dim=1)[1]).float()
    if avg:
        adv_ok = adv_ok.mean()
    return adv_loss, adv_ok


def train(device, writer, epoch, args, prev_layer_idx, curr_layer_idx, next_layer_idx, net, eps_sched, kappa_sched, opt, train_loader, lr_scheduler,
          relu_stable=None):
    net.train()
    train_nat_loss, train_stable_loss, train_nat_ok, train_padv_loss, train_padv_ok, train_adv_loss, train_adv_ok, n_batches = 0, 0, 0, 0, 0, 0, 0, 0
    train_cross_relu = 0
    pbar = tqdm(train_loader, dynamic_ncols=True)

    for i, layer in enumerate(net.blocks[:curr_layer_idx+1]):
        for param in layer.parameters():
            param.requires_grad_(False)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        opt.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        eps, kappa = eps_sched.get(), kappa_sched.get()
        nat_outs = net(inputs)
        nat_loss = F.cross_entropy(nat_outs, targets)
        nat_ok = targets.eq(nat_outs.max(dim=1)[1]).float().mean()
        train_nat_loss += nat_loss.item()
        train_nat_ok += nat_ok.item()
        for param_group in opt.param_groups:
            curr_lr = param_group['lr']

        net.eval()
        if next_layer_idx is not None and relu_stable is not None:
            bounds = compute_bounds_approx(eps, net.blocks, next_layer_idx-1, inputs, k=args.n_rand_proj)
        else:
            with torch.no_grad():
                bounds = compute_bounds_approx(eps, net.blocks, curr_layer_idx, inputs, k=args.n_rand_proj)

        adv_loss, adv_ok = get_adv_loss(device, eps, curr_layer_idx, net, bounds, inputs, targets,
                                        args.train_att_n_steps, args.train_att_step_size, is_train=True)
        train_adv_loss += adv_loss.item()
        train_adv_ok += adv_ok.item()
        if prev_layer_idx == -2:
            padv_loss, padv_ok = nat_loss, nat_ok
        else:
            padv_loss, padv_ok = get_adv_loss(device, eps, prev_layer_idx, net, bounds, inputs, targets,
                                              args.train_att_n_steps, args.train_att_step_size, is_train=True)
        net.train()

        train_padv_loss += padv_loss.item()
        train_padv_ok += padv_ok.item()

        l1_loss = args.l1_reg * compute_l1_loss(net)
        tot_loss = nat_loss * args.nat_factor + (1 - args.nat_factor) * (kappa * adv_loss + (1 - kappa) * padv_loss) + l1_loss
        if next_layer_idx is not None and relu_stable is not None:
            next_lb, next_ub = bounds[next_layer_idx]
            is_cross = (next_lb < 0) & (next_ub > 0)
            train_cross_relu += is_cross.float().sum() / args.train_batch
            stable_loss = (torch.clamp(-next_lb, min=0) * torch.clamp(next_ub, min=0)).sum() / args.train_batch
            tot_loss += kappa * relu_stable * stable_loss
            train_stable_loss += stable_loss.item()
        opt.zero_grad()
        tot_loss.backward()
        opt.step()
        if isinstance(lr_scheduler, optim.lr_scheduler.OneCycleLR):
            lr_scheduler.step()

        n_batches += 1
        pbar.set_description('[T] epoch=%d, nat_ok=%.4f, adv_ok=%.4f, adv_loss=%.4f, cross_relu=%.4f' % (
            epoch,
            train_nat_ok/n_batches,
            train_adv_ok/n_batches,
            train_adv_loss/n_batches,
            train_cross_relu/n_batches,
        ))
        eps_sched.advance_time(args.train_batch)
        kappa_sched.advance_time(args.train_batch)


def run(args):
    device = 'cuda' if torch.cuda.is_available() and (not args.no_cuda) else 'cpu'

    num_train, train_loader, test_loader, input_size, input_channel, n_class = get_loaders(args)
    net = get_network(device, args, input_size, input_channel, n_class)
    print(net)
    n_params = 0
    for param_name, param_value in net.named_parameters():
        if 'deepz_lambda' not in param_name:
            n_params += param_value.numel()
            param_value.requires_grad_(True)
        else:
            param_value.data = torch.ones(param_value.size()).to(device)
            param_value.requires_grad_(False)
    print('Number of parameters: ', n_params)

    n_epochs = args.n_epochs
    if args.train_mode == 'train':
        timestamp = int(time.time())
        model_dir = args.root_dir + 'models_new/%s/%s/%d/%s_%.5f/%d' % (args.dataset, args.exp_name, args.exp_id, args.net, args.train_eps, timestamp)
        print('Saving model to:', model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        args_file = os.path.join(model_dir, 'args.json')
        with open(args_file, 'w') as fou:
            json.dump(vars(args), fou, indent=4)
        writer = None

        epoch = 0
        relu_stable = args.relu_stable
        lr = args.lr
        for j in range(len(args.layers)-1):
            if args.opt == 'adam':
                opt = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
            else:
                opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0)

            if args.lr_sched == 'step_lr':
                lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_factor)
            else:
                lr_scheduler = optim.lr_scheduler.OneCycleLR(opt, div_factor=10000, max_lr=lr, pct_start=args.pct_start,
                                                             steps_per_epoch=len(train_loader), epochs=n_epochs)

            eps = args.eps_factor ** (len(args.layers)-2-j) * (args.start_eps_factor * args.train_eps)
            kappa_sched = Scheduler(0.0, 1.0, num_train * args.mix_epochs, 0)
            eps_sched = Scheduler(0 if args.anneal else eps, eps, num_train * args.mix_epochs, 0)
            prev_layer_idx, curr_layer_idx = args.layers[j], args.layers[j+1]
            next_layer_idx = args.layers[j+2] if j+2 < len(args.layers) else None
            print('new train phase: eps={}, lr={}, prev_layer={}, curr_layer={}, next_layer={}'.format(eps, lr, prev_layer_idx, curr_layer_idx, next_layer_idx))
            layer_dir = '{}/{}'.format(model_dir, curr_layer_idx)
            if not os.path.exists(layer_dir):
                os.makedirs(layer_dir)
            for curr_epoch in range(n_epochs):
                train(device, writer, epoch, args, prev_layer_idx, curr_layer_idx, next_layer_idx, net, eps_sched, kappa_sched, opt, train_loader,
                      lr_scheduler, relu_stable)
                if curr_epoch >= args.mix_epochs and isinstance(lr_scheduler, optim.lr_scheduler.StepLR):
                    lr_scheduler.step()
                if (epoch+1) % args.test_freq == 0:
                    torch.save(net.state_dict(), os.path.join(layer_dir, 'net_%d.pt' % (epoch+1)))
                    with torch.no_grad():
                        valid_nat_loss, valid_nat_acc, valid_robust_loss, valid_robust_acc = test(device, epoch, args, net, test_loader, [curr_layer_idx])
                epoch += 1
            relu_stable = None if relu_stable is None else relu_stable * args.relu_stable_factor
            n_epochs -= args.n_epochs_reduce
            lr = lr * args.lr_layer_dec
    elif args.train_mode == 'print':
        print('printing network to:', args.out_net_file)
        dummy_input = torch.randn(1, input_channel, input_size, input_size, device='cuda')
        net.skip_norm = True
        torch.onnx.export(net, dummy_input, args.out_net_file, verbose=True)
    elif args.train_mode == 'test':
        with torch.no_grad():
            test(device, 0, args, net, test_loader, args.layers)
    else:
        assert False, 'Unknown mode: {}!'.format(args.train_mode)
    return valid_nat_loss, valid_nat_acc, valid_robust_loss, valid_robust_acc


def main():
    args = get_args()
    run(args)


if __name__ == '__main__':
    main()
