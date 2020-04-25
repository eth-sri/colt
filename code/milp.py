import argparse
import numpy as np
import os
import pickle
import time
import torch
import torch.optim as optim
from loaders import get_loaders
from utils import get_inputs, get_network
from gurobipy import GRB, Model, LinExpr
from layers import Linear, ReLU, AveragePooling, Flatten, Conv2d
from main import test, get_adv_loss, compute_bounds, compute_bounds_approx
from tqdm import tqdm

torch.set_printoptions(precision=10)
np.random.seed(100)

BINARY_THRESHOLD = 0.0
dtype = torch.float32
device = 'cuda'


def add_relu_constraints(model, in_lb, in_ub, in_neuron, out_neuron, is_binary):
    if in_ub <= 0:
        out_neuron.lb = 0
        out_neuron.ub = 0
    elif in_lb >= 0:
        model.addConstr(in_neuron, GRB.EQUAL, out_neuron)
    else:
        model.addConstr(out_neuron >= 0)
        model.addConstr(out_neuron >= in_neuron)
        if is_binary:
            relu_ind = model.addVar(vtype=GRB.BINARY)
            model.addConstr(out_neuron <= in_ub * relu_ind)
            model.addConstr(out_neuron <= in_neuron - in_lb * (1 - relu_ind))
            model.addGenConstrIndicator(relu_ind, True, in_neuron, GRB.GREATER_EQUAL, 0.0)
            model.addGenConstrIndicator(relu_ind, False, in_neuron, GRB.LESS_EQUAL, 0.0)
        else:
            model.addConstr(-in_ub * in_neuron + (in_ub - in_lb) * out_neuron, GRB.LESS_EQUAL, -in_lb * in_ub)


def handle_relu(max_binary, lidx, relu_rnk, relu_priority, model, neurons, relu_inds, n_outs, pr_lb, pr_ub, lp=False):
    unstable, n_binary = 0, 0
    neurons[lidx] = []
    relu_inds[lidx] = {}

    binary_left = n_outs if max_binary is None else max_binary
    if relu_rnk is not None and max_binary is not None:
        to_bin = {i: False for i in range(n_outs)}
        for i in range(min(max_binary, n_outs)):
            to_bin[relu_rnk[lidx][i]] = True

    for out_idx in range(n_outs):
        if pr_ub[0, out_idx] <= 0:
            neurons[lidx] += [model.addVar(0, 0, vtype=GRB.CONTINUOUS, name='n_{}_{}'.format(lidx, out_idx))]
        elif pr_lb[0, out_idx] >= 0:
            neurons[lidx] += [neurons[lidx-1][out_idx]]
        else:
            neurons[lidx] += [model.addVar(0, pr_ub[0, out_idx], vtype=GRB.CONTINUOUS, name='n_{}_{}'.format(lidx, out_idx))]
            model.addConstr(neurons[lidx][out_idx] >= 0)
            model.addConstr(neurons[lidx][out_idx] >= neurons[lidx-1][out_idx])
            
            if lp:
                is_binary = False
            elif max_binary is None:
                is_binary = True
            else:
                is_binary = to_bin[out_idx] if relu_rnk is not None else binary_left > 0
            if is_binary:
                binary_left -= 1
                n_binary += 1
                relu_inds[lidx][out_idx] = model.addVar(vtype=GRB.BINARY, name='ind_relu_{}_{}'.format(lidx, out_idx))
                relu_inds[lidx][out_idx].BranchPriority = relu_priority[lidx][out_idx]
                model.addConstr(neurons[lidx][out_idx] <= pr_ub[0, out_idx] * relu_inds[lidx][out_idx])
                model.addConstr(neurons[lidx][out_idx] <= neurons[lidx-1][out_idx] - pr_lb[0, out_idx] * (1 - relu_inds[lidx][out_idx]))
                model.addGenConstrIndicator(relu_inds[lidx][out_idx], True, neurons[lidx-1][out_idx], GRB.GREATER_EQUAL, 0.0)
                model.addGenConstrIndicator(relu_inds[lidx][out_idx], False, neurons[lidx-1][out_idx], GRB.LESS_EQUAL, 0.0)
            else:
                model.addConstr(
                    -pr_ub[0, out_idx] * neurons[lidx-1][out_idx] + (pr_ub[0, out_idx] - pr_lb[0, out_idx]) * neurons[lidx][out_idx],
                    GRB.LESS_EQUAL,
                    -pr_lb[0, out_idx] * pr_ub[0, out_idx])
            unstable += 1
    return unstable, n_binary


def report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data):
    if tot_tests % 10 == 0:
        print('tot_tests: %d, verified: %.5lf [%d/%d], nat_ok: %.5lf [%d/%d], attack_ok: %.5lf [%d/%d], pgd_ok: %.5lf [%d/%d]' % (
            tot_tests,
            tot_verified_corr/tot_tests, tot_verified_corr, tot_tests,
            tot_nat_ok/tot_tests, tot_nat_ok, tot_tests,
            tot_attack_ok/tot_tests, tot_attack_ok, tot_tests,
            tot_pgd_ok/tot_tests, tot_pgd_ok, tot_tests,
        ))
        print('=====================================')
    out_file = os.path.join(ver_logdir, '{}.p'.format(test_idx))
    pickle.dump(test_data, open(out_file, 'wb'))


def get_diff(outs, targets, i, j, reduction):
    assert reduction == 'sum'
    return -(outs[:, i] - outs[:, ]).sum()


def learn_slopes(relu_params, bounds, args, n_layers, net, inputs, targets, abs_inputs, i, j):
    for param in relu_params:
        param.data = torch.ones(param.size()).to(param.device)
    relu_opt = optim.Adam(relu_params, lr=0.03, weight_decay=0)
    ret_verified = False

    for it in range(args.num_iters):
        relu_opt.zero_grad()
        if i is None and j is None:
            abs_loss, abs_ok = get_adv_loss(device, args.test_eps, args.layer_idx, net, bounds, inputs, targets, args, detach=False)
        else:
            abs_out = net(abs_inputs)
            abs_loss = -abs_out.get_min_diff(i, j)
        relu_opt.zero_grad()
        abs_loss.backward()
        relu_opt.step()
        for param in relu_params:
            if param.grad is not None:
                param.data = torch.clamp(param.data, 0, 1)
        if ret_verified:
            break

    with torch.no_grad():
        abs_out = net(abs_inputs)
        if i is None and j is None:
            _, verified_corr = abs_out.verify(targets)
            if verified_corr:
                ret_verified = True
        else:
            abs_loss = -abs_out.get_min_diff(i, j)
            if abs_loss < 0:
                ret_verified = True

    relu_rnk, relu_priority = {}, {}
    for lidx, layer in enumerate(net.blocks):
        if isinstance(layer, ReLU):
            for param in layer.parameters():
                relu_priority[lidx] = []

                if param.grad is None:
                    for i in range(param.size()[0]):
                        relu_priority[lidx].append(0)
                    _, sorted_ids = torch.sort(param.abs().view(-1), descending=True)
                else:
                    g_abs = param.grad.abs().view(-1)
                    for i in range(g_abs.size()[0]):
                        relu_priority[lidx].append(int(g_abs[i].item()*1000))
                    _, sorted_ids = torch.sort(param.grad.abs().view(-1), descending=True)
                sorted_ids = sorted_ids.cpu().numpy()
                relu_rnk[lidx] = sorted_ids
    net.zero_grad()
    return relu_rnk, ret_verified, relu_priority


def callback(model, where):
    if where == GRB.Callback.MIP:
        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        if obj_bound > 0 or obj_best < 0:
            model.terminate()


def reset_params(args, net, dtype):
    relu_params = []
    for param_name, param_value in net.named_parameters():
        if 'deepz' in param_name:
            relu_params.append(param_value)
            if args.test_domain == 'zono_iter':
                param_value.data = torch.ones(param_value.size()).to(param_value.device, dtype=dtype)
            else:
                param_value.data = torch.ones(param_value.size()).to(param_value.device, dtype=dtype)
            param_value.requires_grad_(True)
        else:
            param_value.requires_grad_(False)
    return relu_params


def get_flat_idx(img_dim, ch_idx, i, j):
    return ch_idx * (img_dim) ** 2 + i * img_dim + j


def refine(args, bounds, net, refine_i, refine_j, abs_inputs, input_size):
    dep = {lidx: {} for lidx in range(-1, args.refine_lidx+1)}
    neurons = {lidx: {} for lidx in range(-1, args.refine_lidx+1)}
    dep[args.refine_lidx][(refine_i, refine_j)] = True
    model = Model("refinezono")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 10)
    to_refine = []
    refine_channels = bounds[args.refine_lidx+1][0].shape[1]
    for ch_idx in range(refine_channels):
        out_lb = bounds[args.refine_lidx+1][0][0, ch_idx, refine_i, refine_j]
        out_ub = bounds[args.refine_lidx+1][1][0, ch_idx, refine_i, refine_j]
        neurons[args.refine_lidx][(ch_idx, refine_i, refine_j)] = model.addVar(out_lb, out_ub, vtype=GRB.CONTINUOUS)
        if out_lb < 0 and out_ub > 0:
            to_refine.append(ch_idx)

    binary_left = args.refine_milp
    for lidx in range(args.refine_lidx, 0, -1):
        lb, ub = bounds[lidx]
        block = net.blocks[lidx]
        if isinstance(block, Conv2d):
            weight, bias = block.conv.weight.cpu().numpy(), block.conv.bias.cpu().numpy()
            kernel_size, stride = block.kernel_size, block.stride
            dim = bounds[lidx+1][0].shape[2]
            out_channels, in_channels = weight.shape[0], weight.shape[1]
            if kernel_size % 2 == 0:
                min_kdelta, max_kdelta = -(kernel_size//2-1), kernel_size//2
            else:
                min_kdelta, max_kdelta = -(kernel_size//2), kernel_size//2

            for x in range(0, dim):
                for y in range(0, dim):
                    if (x, y) not in dep[lidx]:
                        continue
                    for out_ch in range(out_channels):
                        expr = LinExpr()
                        expr += bias[out_ch]
                        for kx in range(min_kdelta, max_kdelta+1):
                            for ky in range(min_kdelta, max_kdelta+1):
                                new_x = x*stride + kx
                                new_y = y*stride + ky
                                if new_x < 0 or new_y < 0 or new_x >= dim*stride or new_y >= dim*stride:
                                    continue
                                dep[lidx-1][(new_x, new_y)] = True
                                for in_ch in range(in_channels):
                                    if (in_ch, new_x, new_y) not in neurons[lidx-1]:
                                        in_lb = bounds[lidx][0][0, in_ch, new_x, new_y].item()
                                        in_ub = bounds[lidx][1][0, in_ch, new_x, new_y].item()
                                        neurons[lidx-1][(in_ch, new_x, new_y)] = model.addVar(in_lb, in_ub, vtype=GRB.CONTINUOUS)
                                    expr += neurons[lidx-1][(in_ch, new_x, new_y)] * weight[out_ch, in_ch, kx - min_kdelta, ky - min_kdelta]
                        model.addConstr(expr, GRB.EQUAL, neurons[lidx][(out_ch, x, y)])
        elif isinstance(block, ReLU):
            n_channels, dim = lb.shape[1], lb.shape[2]
            for x in range(0, dim):
                for y in range(0, dim):
                    if (x, y) not in dep[lidx]:
                        continue
                    dep[lidx-1][(x, y)] = True
                    for out_ch in range(n_channels):
                        in_lb, in_ub = lb[0, out_ch, x, y].item(), ub[0, out_ch, x, y].item()
                        neurons[lidx-1][(out_ch, x, y)] = model.addVar(in_lb, in_ub, vtype=GRB.CONTINUOUS)
                        if binary_left > 0 and in_lb < 0 and in_ub > 0:
                            is_binary = True
                            binary_left -= 1
                        else:
                            is_binary = False
                        add_relu_constraints(model, in_lb, in_ub, neurons[lidx-1][(out_ch, x, y)], neurons[lidx][(out_ch, x, y)], is_binary)
        else:
            assert False

    for ch_idx in to_refine:
        old_lb = bounds[args.refine_lidx+1][0][0, ch_idx, refine_i, refine_j]
        old_ub = bounds[args.refine_lidx+1][1][0, ch_idx, refine_i, refine_j]
        model.setObjective(neurons[args.refine_lidx][(ch_idx, refine_i, refine_j)], GRB.MINIMIZE)
        model.update()
        model.optimize()
        new_lb = model.objBound
        model.setObjective(neurons[args.refine_lidx][(ch_idx, refine_i, refine_j)], GRB.MAXIMIZE)
        model.update()
        model.optimize()
        new_ub = model.objBound
        if new_lb != -GRB.INFINITY and new_lb >= old_lb:
            net.blocks[args.refine_lidx+1].bounds[0][0, ch_idx, refine_i, refine_j] = new_lb
        if new_ub != GRB.INFINITY and new_ub <= old_ub:
            net.blocks[args.refine_lidx+1].bounds[1][0, ch_idx, refine_i, refine_j] = new_ub


def verify_test(args, net, num_relu, inputs, targets, abs_inputs, bounds, refined_triples, test_data, grb_modelsdir, test_idx):
    ok = True
    model = None
    n_layers = len(net.blocks)
    for adv_idx in range(10):
        if targets[0] == adv_idx:
            continue
        if ('verified', adv_idx) in test_data and test_data[('verified', adv_idx)]:
            print('label already verified: ', adv_idx)
            continue
        relu_params = reset_params(args, net, dtype)
        if adv_idx in test_data:
            print(test_data[adv_idx])
        if args.obj_threshold is not None:
            if adv_idx not in test_data:
                ok = False
                continue
            if test_data[adv_idx]['obj_bound'] < args.obj_threshold:
                print('too far, not considering adv_idx = %d, obj_bound = %.5lf' % (adv_idx, test_data[adv_idx]['obj_bound']))
                ok = False
                continue

        relu_rnk, relu_priority = None, None
        if args.test_domain == 'zono_iter':
            with torch.enable_grad():
                relu_rnk, verified, relu_priority = learn_slopes(
                    relu_params, bounds, args, n_layers, net, inputs, targets, abs_inputs, targets[0].item(), adv_idx)
                if verified:
                    print('adv_idx=%d verified without MILP' % adv_idx)
                    test_data[('verified', adv_idx)] = True
                    continue

        max_binary = args.max_binary
        milp_timeout = args.milp_timeout

        if model is None or (args.test_domain == 'zono_iter'):
            model = Model("milp")
            model.setParam('OutputFlag', args.debug)
            model.setParam('TimeLimit', milp_timeout)

            abs_curr = net.forward_until(args.layer_idx, abs_inputs)
            abs_flat = abs_curr
            
            if len(abs_curr.head.size()) == 4:
                n_channels, img_dim = abs_curr.head.size()[1], abs_curr.head.size()[2]
                flat_dim = n_channels * img_dim * img_dim
                abs_flat = abs_curr.view((1, flat_dim))

            n_inputs = abs_flat.head.size()[1]
            betas = [model.addVar(-1.0, 1.0, vtype=GRB.CONTINUOUS, name='beta_{}'.format(j)) for j in range(n_inputs)]
            if abs_flat.errors is not None:
                n_errors = abs_flat.errors.size()[0]
                errors = [model.addVar(-1.0, 1.0, vtype=GRB.CONTINUOUS, name='error_{}'.format(j)) for j in range(n_errors)]
                
            if net.blocks[args.layer_idx+1].bounds is not None:
                lb_refine, ub_refine = net.blocks[args.layer_idx+1].bounds
                lb_refine, ub_refine = lb_refine.view((1, -1)).cpu().numpy(), ub_refine.view((1, -1)).cpu().numpy()
                
            lb, ub = abs_flat.concretize()
            lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()
            neurons, relu_inds = {}, {}
            neurons[args.layer_idx] = []
            for j in range(n_inputs):
                neuron_lb, neuron_ub = lb[0, j], ub[0, j]
                if net.blocks[args.layer_idx+1].bounds is not None:
                    neuron_lb = lb_refine[0, j]
                    neuron_ub = ub_refine[0, j]
                    lb[0, j] = neuron_lb
                    ub[0, j] = neuron_ub
                neurons[args.layer_idx].append(model.addVar(vtype=GRB.CONTINUOUS, lb=neuron_lb, ub=neuron_ub, name='input_{}'.format(j)))
                expr = LinExpr()
                expr += abs_flat.head[0, j].item()
                if abs_flat.beta is not None:
                    expr += abs_flat.beta[0, j].item() * betas[j]
                if abs_flat.errors is not None:
                    coeffs = abs_flat.errors[:, 0, j].detach().cpu().numpy().tolist()
                    expr += LinExpr(coeffs, errors)
                model.addConstr(expr, GRB.EQUAL, neurons[args.layer_idx][j])
            n_outs = n_inputs

            relu_done = False
            for lidx in range(args.layer_idx+1, n_layers):
                pr_lb, pr_ub = lb, ub
                abs_curr = net.blocks[lidx](abs_curr)
                if len(abs_curr.head.size()) == 4:
                    n_channels, img_dim = abs_curr.head.size()[1], abs_curr.head.size()[2]
                    flat_dim = n_channels * img_dim * img_dim
                    abs_flat = abs_curr.view((1, flat_dim))
                else:
                    abs_flat = abs_curr

                lb, ub = abs_flat.concretize()
                lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()
                if isinstance(net.blocks[lidx], Linear):
                    weight, bias = net.blocks[lidx].linear.weight, net.blocks[lidx].linear.bias
                    neurons[lidx] = []
                    n_outs = weight.size()[0]

                    for out_idx in range(n_outs):
                        nvar = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[0, out_idx], ub=ub[0, out_idx], name='n_{}_{}'.format(lidx, out_idx))
                        neurons[lidx].append(nvar)
                        tmp = LinExpr()
                        tmp += -neurons[lidx][out_idx]
                        tmp += bias[out_idx].item()
                        tmp += LinExpr(weight[out_idx].detach().cpu().numpy(), neurons[lidx-1])
                        model.addConstr(tmp, GRB.EQUAL, 0)
                elif isinstance(net.blocks[lidx], ReLU):
                    lp = False
                    unstable, n_binary = handle_relu(
                        max_binary, lidx, relu_rnk, relu_priority, model, neurons, relu_inds, n_outs, pr_lb, pr_ub, lp)
                    relu_done = True
                    print('Unstable ReLU: ', unstable, ' binary: ', n_binary)
                elif isinstance(net.blocks[lidx], AveragePooling):
                    kernel_size = net.blocks[lidx].kernel_size
                    assert img_dim % kernel_size == 0
                    neurons[lidx] = []
                    for ch_idx in range(n_channels):
                        for i in range(0, img_dim, kernel_size):
                            for j in range(0, img_dim, kernel_size):
                                new_idx = get_flat_idx(img_dim//kernel_size, ch_idx, i//kernel_size, j//kernel_size)
                                nvar = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[0, new_idx], ub=ub[0, new_idx], name='n_{}_{}'.format(lidx, new_idx))
                                neurons[lidx].append(nvar)

                                tmp = LinExpr()
                                tmp -= (kernel_size * kernel_size) * nvar
                                for di in range(0, kernel_size):
                                    for dj in range(0, kernel_size):
                                        old_idx = get_flat_idx(img_dim, ch_idx, i+di, j+dj)
                                        tmp += neurons[lidx-1][old_idx]
                                model.addConstr(tmp, GRB.EQUAL, 0)
                elif isinstance(net.blocks[lidx], Flatten):
                    neurons[lidx] = neurons[lidx-1]
                elif isinstance(net.blocks[lidx], Conv2d):
                    weight, bias = net.blocks[lidx].conv.weight.cpu().numpy(), net.blocks[lidx].conv.bias.cpu().numpy()
                    kernel_size, stride = net.blocks[lidx].kernel_size, net.blocks[lidx].stride
                    out_channels, in_channels = weight.shape[0], weight.shape[1]
                    if kernel_size % 2 == 0:
                        min_kdelta, max_kdelta = -(kernel_size//2-1), kernel_size//2
                    else:
                        min_kdelta, max_kdelta = -(kernel_size//2), kernel_size//2

                    neurons[lidx] = []
                    for out_ch in range(out_channels):
                        for x in range(0, img_dim):
                            for y in range(0, img_dim):
                                new_idx = get_flat_idx(img_dim, out_ch, x, y)
                                nvar = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[0, new_idx], ub=ub[0, new_idx], name='n_{}_{}'.format(lidx, new_idx))
                                neurons[lidx].append(nvar)
                                expr = LinExpr()
                                expr += bias[out_ch]
                                for kx in range(min_kdelta, max_kdelta+1):
                                    for ky in range(min_kdelta, max_kdelta+1):
                                        new_x = x*stride + kx
                                        new_y = y*stride + ky
                                        if new_x < 0 or new_y < 0 or new_x >= img_dim*stride or new_y >= img_dim*stride:
                                            continue
                                        for in_ch in range(in_channels):
                                            old_idx = get_flat_idx(img_dim*stride, in_ch, new_x, new_y)
                                            expr += neurons[lidx-1][old_idx] * weight[out_ch, in_ch, kx - min_kdelta, ky - min_kdelta]
                                model.addConstr(expr, GRB.EQUAL, nvar)
                else:
                    print('unknown layer type: ', net.blocks[lidx])
                    assert False

        model.setObjective(neurons[n_layers-1][targets[0].item()] - neurons[n_layers-1][adv_idx], GRB.MINIMIZE)
        model.update()

        if args.save_models:
            model.write('%s/model_%d_%d.mps' % (grb_modelsdir, test_idx, adv_idx))
            ok = False
            continue
        
        model.optimize(callback)
        print('MILP: ', targets[0].item(), adv_idx, model.objVal, model.objBound, model.RunTime)
        test_data[adv_idx] = {
            'milp_timeout': milp_timeout,
            'max_binary': max_binary,
            'obj_val': model.objVal,
            'obj_bound': model.objBound,
            'runtime': model.RunTime,
        }
        if model.objBound < 0:
            ok = False
            break
        else:
            test_data[('verified', adv_idx)] = True
    return ok


def main():
    parser = argparse.ArgumentParser(description='Perform greedy layerwise training.')
    parser.add_argument('--prune_p', default=None, type=float, help='percentage of weights to prune in each layer')
    parser.add_argument('--dataset', default='cifar10', help='dataset to use')
    parser.add_argument('--net', required=True, type=str, help='network to use')
    parser.add_argument('--load_model', type=str, help='model to load')
    parser.add_argument('--layer_idx', default=1, type=int, help='layer index of flattened vector')
    parser.add_argument('--n_valid', default=1000, type=int, help='number of test samples')
    parser.add_argument('--n_train', default=None, type=int, help='number of training samples to use')
    parser.add_argument('--train_batch', default=1, type=int, help='batch size for training')
    parser.add_argument('--test_batch', default=128, type=int, help='batch size for testing')
    parser.add_argument('--test_domain', default='zono', type=str, help='domain to test with')
    parser.add_argument('--test_eps', default=None, type=float, help='epsilon to verify')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--no_milp', action='store_true', help='no MILP mode')
    parser.add_argument('--no_load', action='store_true', help='verify from scratch')
    parser.add_argument('--no_smart', action='store_true', help='bla')
    parser.add_argument('--milp_timeout', default=10, type=int, help='timeout for MILP')
    parser.add_argument('--eval_train', action='store_true', help='evaluate on training set')
    parser.add_argument('--test_idx', default=None, type=int, help='specific index to test')
    parser.add_argument('--start_idx', default=0, type=int, help='specific index to start')
    parser.add_argument('--end_idx', default=1000, type=int, help='specific index to end')
    parser.add_argument('--max_binary', default=None, type=int, help='number of neurons to encode as binary variable in MILP (per layer)')
    parser.add_argument('--num_iters', default=50, type=int, help='number of iterations to find slopes')
    parser.add_argument('--max_refine_triples', default=0, type=int, help='number of triples to refine')
    parser.add_argument('--refine_lidx', default=None, type=int, help='layer to refine')
    parser.add_argument('--save_models', action='store_true', help='whether to only store models')
    parser.add_argument('--refine_milp', default=0, type=int, help='number of neurons to refine using MILP')
    parser.add_argument('--obj_threshold', default=None, type=float, help='threshold to consider for MILP verification')
    parser.add_argument('--attack_type', default='pgd', type=str, help='attack')
    parser.add_argument('--attack_n_steps', default=10, type=int, help='number of steps for the attack')
    parser.add_argument('--attack_step_size', default=0.25, type=float, help='step size for the attack (relative to epsilon)')
    parser.add_argument('--layers', required=False, default=None, type=int, nargs='+', help='layer indices for training')
    args = parser.parse_args()

    ver_logdir = args.load_model[:-3] + '_ver'
    if not os.path.exists(ver_logdir):
        os.makedirs(ver_logdir)
    grb_modelsdir = args.load_model[:-3] + '_grb'
    if not os.path.exists(grb_modelsdir):
        os.makedirs(grb_modelsdir)

    num_train, _, test_loader, input_size, input_channel = get_loaders(args)
    net = get_network(device, args, input_size, input_channel)
    n_layers = len(net.blocks)
    
    # net.to_double()

    args.test_domains = ['box']
    with torch.no_grad():
        test(device, 0, args, net, test_loader)

    args.test_batch = 1
    num_train, _, test_loader, input_size, input_channel = get_loaders(args)

    num_relu = 0
    for lidx in range(args.layer_idx+1, n_layers):
        print(net.blocks[lidx])
        if isinstance(net.blocks[lidx], ReLU):
            num_relu += 1

    with torch.no_grad():
        tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, tot_tests = 0, 0, 0, 0, 0
        for test_idx, (inputs, targets) in enumerate(test_loader):
            if test_idx < args.start_idx or test_idx >= args.end_idx or test_idx >= args.n_valid:
                continue
            if args.test_idx is not None and test_idx != args.test_idx:
                continue
            tot_tests += 1
            test_file = os.path.join(ver_logdir, '{}.p'.format(test_idx))
            test_data = pickle.load(open(test_file, 'rb')) if (not args.no_load) and os.path.isfile(test_file) else {}
            print('Verify test_idx =', test_idx)

            for lidx in range(n_layers):
                net.blocks[lidx].bounds = None

            inputs, targets = inputs.to(device), targets.to(device)
            abs_inputs = get_inputs(args.test_domain, inputs, args.test_eps, device, dtype=dtype)
            nat_out = net(inputs)
            nat_ok = targets.eq(nat_out.max(dim=1)[1]).item()
            tot_nat_ok += float(nat_ok)
            test_data['ok'] = nat_ok
            if not nat_ok:
                report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
                continue

            with torch.enable_grad():
                pgd_loss, pgd_ok = get_adv_loss(device, args.test_eps, -1, net, None, inputs, targets, args)
            if pgd_ok:
                test_data['pgd_ok'] = 1
                tot_pgd_ok += 1
            else:
                test_data['pgd_ok'] = 0
                report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
                continue
            if 'verified' in test_data and test_data['verified']:
                tot_verified_corr += 1
                tot_attack_ok += 1
                report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
                continue

            relu_params = reset_params(args, net, dtype)

            bounds = compute_bounds(net, device, args.layer_idx, args, abs_inputs)
            if args.test_domain == 'zono_iter':
                with torch.enable_grad():
                    learn_slopes(relu_params, bounds, args, n_layers, net, inputs, targets, abs_inputs, None, None)

            with torch.enable_grad():
                abs_loss, abs_ok = get_adv_loss(device, args.test_eps, args.layer_idx, net, bounds, inputs, targets, args)

            refined_triples = []
            if args.refine_lidx is not None:
                bounds = compute_bounds(net, device, args.layer_idx+1, args, abs_inputs)
                for lidx in range(0, args.layer_idx+2):
                    net.blocks[lidx].bounds = bounds[lidx]
                print('loss before refine: ', abs_loss)
                refine_dim = bounds[args.refine_lidx+1][0].shape[2]
                pbar = tqdm(total=refine_dim*refine_dim, dynamic_ncols=True)
                for refine_i in range(refine_dim):
                    for refine_j in range(refine_dim):
                        # refine(args, bounds, net, 0, 15, abs_inputs, input_size)
                        refine(args, bounds, net, refine_i, refine_j, abs_inputs, input_size)
                        pbar.update(1)
                pbar.close()
                with torch.enable_grad():
                    abs_loss, abs_ok = get_adv_loss(device, args.test_eps, args.layer_idx, net, bounds, inputs, targets, args)
                print('loss after refine: ', abs_loss)

            if abs_ok:
                tot_attack_ok += 1
            abs_out = net(abs_inputs)
            verified, verified_corr = abs_out.verify(targets)
            test_data['verified'] = int(verified_corr.item())
            print('abs_loss: ', abs_loss.item(), '\tabs_ok: ', abs_ok.item(), '\tverified_corr: ', verified_corr.item())
            if verified_corr:
                tot_verified_corr += 1
                report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
                continue
            if args.no_milp or (not abs_ok):
                report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
                continue

            if verify_test(args, net, num_relu, inputs, targets, abs_inputs, bounds, refined_triples, test_data, grb_modelsdir, test_idx):
                tot_verified_corr += 1
                test_data['verified'] = True
            report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data)


if __name__ == '__main__':
    main()
