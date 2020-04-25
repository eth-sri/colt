import sys
sys.path.append('../')
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from args_factory_verifier import get_args
from gurobipy import GRB, Model, LinExpr
from layers import Linear, ReLU, Flatten, Conv2d
from loaders import get_loaders
from learner import learn_slopes, learn_bounds
from main import test, get_adv_loss, compute_bounds
from utils import get_inputs, get_network
from refinement import refine

dtype = torch.float64
device = 'cuda'


def report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data):
    """ Logs evaluation statistics to standard output. """
    if tot_tests % 1 == 0:
        print('tot_tests: %d, verified: %.5lf [%d/%d], nat_ok: %.5lf [%d/%d], latent_ok: %.5lf [%d/%d], pgd_ok: %.5lf [%d/%d]' % (
            tot_tests,
            tot_verified_corr/tot_tests, tot_verified_corr, tot_tests,
            tot_nat_ok/tot_tests, tot_nat_ok, tot_tests,
            tot_attack_ok/tot_tests, tot_attack_ok, tot_tests,
            tot_pgd_ok/tot_tests, tot_pgd_ok, tot_tests,
        ))
        print('=====================================')
    out_file = os.path.join(ver_logdir, '{}.p'.format(test_idx))
    pickle.dump(test_data, open(out_file, 'wb'))


def reset_params(args, net, dtype):
    """ Resets DeepZ slope parameters to the original values. """
    relu_params = []
    for param_name, param_value in net.named_parameters():
        if 'deepz' in param_name:
            relu_params.append(param_value)
            if args.test_domain == 'zono_iter':
                param_value.data = torch.ones(param_value.size()).to(param_value.device, dtype=dtype)
            param_value.requires_grad_(True)
        else:
            param_value.requires_grad_(False)
    return relu_params


def callback(model, where):
    if where == GRB.Callback.MIP:
        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        if obj_bound > 0 or obj_best < 0:
            model.terminate()


def handle_relu(max_binary, lidx, relu_priority, model, neurons, n_outs, pr_lb, pr_ub, bin_neurons=None):
    unstable, n_binary = 0, 0
    neurons[lidx] = []
    for out_idx in range(n_outs):
        if pr_ub[0, out_idx] <= 0:
            neurons[lidx] += [model.addVar(0, 0, vtype=GRB.CONTINUOUS, name='n_{}_{}'.format(lidx, out_idx))]
        elif pr_lb[0, out_idx] >= 0:
            neurons[lidx] += [neurons[lidx-1][out_idx]]
        else:
            neurons[lidx] += [model.addVar(0, pr_ub[0, out_idx], vtype=GRB.CONTINUOUS, name='n_{}_{}'.format(lidx, out_idx))]

    binary_left = n_outs if max_binary is None else max_binary

    out_ids = list(range(n_outs))
    out_ids.sort(key=lambda x: -relu_priority[x])

    for out_idx in out_ids:
        if pr_lb[0, out_idx] >= 0 or pr_ub[0, out_idx] <= 0:
            continue
        model.addConstr(neurons[lidx][out_idx] >= 0)
        model.addConstr(neurons[lidx][out_idx] >= neurons[lidx-1][out_idx])

        if binary_left > 0 and ((bin_neurons is None) or (lidx, out_idx) in bin_neurons):
            binary_left -= 1
            n_binary += 1
            relu_ind = model.addVar(vtype=GRB.BINARY, name='ind_relu_{}_{}'.format(lidx, out_idx))
            model.addConstr(neurons[lidx][out_idx] <= pr_ub[0, out_idx] * relu_ind)
            model.addConstr(neurons[lidx][out_idx] <= neurons[lidx-1][out_idx] - pr_lb[0, out_idx] * (1 - relu_ind))
            model.addGenConstrIndicator(relu_ind, True, neurons[lidx-1][out_idx], GRB.GREATER_EQUAL, 0.0)
            model.addGenConstrIndicator(relu_ind, False, neurons[lidx-1][out_idx], GRB.LESS_EQUAL, 0.0)
            relu_ind.BranchPriority = relu_priority[out_idx]
        else:
            model.addConstr(
                -pr_ub[0, out_idx] * neurons[lidx-1][out_idx] + (pr_ub[0, out_idx] - pr_lb[0, out_idx]) * neurons[lidx][out_idx],
                GRB.LESS_EQUAL,
                -pr_lb[0, out_idx] * pr_ub[0, out_idx])
        unstable += 1
    return unstable, n_binary


def verify_test(args, net, inputs, targets, abs_inputs, bounds, test_data, test_idx):
    ok = True
    n_layers = len(net.blocks)
    for adv_idx in range(10):
        if targets[0] == adv_idx:
            continue
        if ('verified', adv_idx) in test_data and test_data[('verified', adv_idx)]:
            print('label already verified: ', adv_idx)
            continue
        relu_params = reset_params(args, net, dtype)
        bin_neurons = None
        with torch.enable_grad():
            verified, relu_priority = learn_slopes(
                device, relu_params, bounds, args, n_layers, net, inputs, targets, abs_inputs, targets[0].item(), adv_idx)
            if args.tot_binary is not None:
                bin_neurons = {}
                all_neurons = []
                for layer_idx, neurons in relu_priority.items():
                    for neuron_idx, neuron_priority in enumerate(neurons):
                        all_neurons += [(layer_idx, neuron_idx, neuron_priority)]
                        # print(layer_idx, neuron_idx, neuron_priority)
                all_neurons.sort(key=lambda x: -x[2])
                bin_neurons = {(layer_idx, neuron_idx): True for layer_idx, neuron_idx, neuron_priority in all_neurons[:args.tot_binary]}
        if verified:
            print('adv_idx=%d verified without MILP' % adv_idx)
            test_data[('verified', adv_idx)] = True
            continue

        model = Model("milp")
        model.setParam('OutputFlag', args.debug)
        model.setParam('TimeLimit', args.milp_timeout)

        abs_curr = net.forward_until(args.layer_idx, abs_inputs)
        abs_flat = abs_curr.view((1, -1))
        n_inputs = abs_flat.head.size()[1]
        if abs_flat.errors is not None:
            n_errors = abs_flat.errors.size()[0]
            errors = [model.addVar(-1.0, 1.0, vtype=GRB.CONTINUOUS, name='error_{}'.format(j)) for j in range(n_errors)]

        lb, ub = abs_flat.concretize()
        lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()
        neurons = {}
        neurons[args.layer_idx] = []
        for j in range(n_inputs):
            neurons[args.layer_idx] += [model.addVar(lb=lb[0, j], ub=ub[0, j], vtype=GRB.CONTINUOUS, name='input_{}'.format(j))]
            expr = LinExpr()
            expr += abs_flat.head[0, j].item()
            expr += LinExpr(abs_flat.errors[:, 0, j].detach().cpu().numpy().tolist(), errors)
            model.addConstr(expr, GRB.EQUAL, neurons[args.layer_idx][j])
        n_outs = n_inputs

        for lidx in range(args.layer_idx+1, n_layers):
            pr_lb, pr_ub = lb, ub
            abs_curr = net.blocks[lidx](abs_curr)
            abs_flat = abs_curr.view((1, -1))

            neurons[lidx] = []
            lb, ub = abs_flat.concretize()
            lb, ub = lb.detach().cpu().numpy(), ub.detach().cpu().numpy()
            if isinstance(net.blocks[lidx], Linear):
                weight, bias = net.blocks[lidx].linear.weight, net.blocks[lidx].linear.bias
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
                if net.blocks[lidx].bounds is not None:
                    pr_lb = np.maximum(pr_lb, net.blocks[lidx].bounds[0].view((1, -1)).cpu().numpy())
                    pr_ub = np.minimum(pr_ub, net.blocks[lidx].bounds[1].view((1, -1)).cpu().numpy())

                unstable, n_binary = handle_relu(args.max_binary, lidx, relu_priority[lidx],
                                                 model, neurons, n_outs, pr_lb, pr_ub, bin_neurons=bin_neurons)
                print('Unstable ReLU: ', unstable, ' binary: ', n_binary)
            elif isinstance(net.blocks[lidx], Flatten):
                # print('flatten')
                neurons[lidx] = neurons[lidx-1]
            elif isinstance(net.blocks[lidx], Conv2d):
                img_dim = abs_curr.head.size()[2]
                weight, bias = net.blocks[lidx].conv.weight.cpu().numpy(), net.blocks[lidx].conv.bias.cpu().numpy()
                kernel_size, stride, pad = net.blocks[lidx].kernel_size, net.blocks[lidx].stride, net.blocks[lidx].padding
                out_channels, in_channels = weight.shape[0], weight.shape[1]

                neurons[lidx] = []
                for out_ch in range(out_channels):
                    for x in range(0, img_dim):
                        for y in range(0, img_dim):
                            new_idx = out_ch * (img_dim ** 2) + x * img_dim + y
                            nvar = model.addVar(vtype=GRB.CONTINUOUS, lb=lb[0, new_idx], ub=ub[0, new_idx], name='n_{}_{}'.format(lidx, new_idx))
                            neurons[lidx].append(nvar)
                            expr = LinExpr()
                            expr += bias[out_ch]
                            for kx in range(0, kernel_size):
                                for ky in range(0, kernel_size):
                                    new_x = -pad + x * stride + kx
                                    new_y = -pad + y * stride + ky
                                    if new_x < 0 or new_y < 0 or new_x >= net.blocks[lidx].bounds.size(2) or new_y >= net.blocks[lidx].bounds.size(3):
                                        continue
                                    for in_ch in range(in_channels):
                                        old_idx = in_ch * (net.blocks[lidx].bounds.size(2) * net.blocks[lidx].bounds.size(3)) \
                                                  + new_x * (net.blocks[lidx].bounds.size(3)) + new_y
                                        expr += neurons[lidx-1][old_idx] * weight[out_ch, in_ch, kx, ky]
                            model.addConstr(expr, GRB.EQUAL, nvar)
            else:
                print('unknown layer type: ', net.blocks[lidx])
                assert False

        model.setObjective(neurons[n_layers-1][targets[0].item()] - neurons[n_layers-1][adv_idx], GRB.MINIMIZE)
        model.update()
        model.optimize(callback)
        
        print('MILP: ', targets[0].item(), adv_idx, model.objVal, model.objBound, model.RunTime)
        test_data[adv_idx] = {
            'milp_timeout': args.milp_timeout,
            'max_binary': args.max_binary,
            'obj_val': model.objVal,
            'obj_bound': model.objBound,
            'runtime': model.RunTime,
        }
        if model.objBound < 0:
            ok = False
            if args.fail_break:
                break
        else:
            test_data[('verified', adv_idx)] = True
    return ok


def main():
    args = get_args()

    ver_logdir = args.load_model[:-3] + '_ver'
    if not os.path.exists(ver_logdir):
        os.makedirs(ver_logdir)

    num_train, _, test_loader, input_size, input_channel, n_class = get_loaders(args)
    net = get_network(device, args, input_size, input_channel, n_class)
    print(net)

    args.test_domains = []
    # with torch.no_grad():
    #     test(device, 0, args, net, test_loader, layers=[-1, args.layer_idx])
    args.test_batch = 1
    num_train, _, test_loader, input_size, input_channel, n_class = get_loaders(args)
    latent_idx = args.layer_idx if args.latent_idx is None else args.latent_idx
    img_file = open(args.unverified_imgs_file, 'w')

    with torch.no_grad():
        tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, tot_tests = 0, 0, 0, 0, 0
        for test_idx, (inputs, targets) in enumerate(test_loader):
            if test_idx < args.start_idx or test_idx >= args.end_idx:
                continue
            tot_tests += 1
            test_file = os.path.join(ver_logdir, '{}.p'.format(test_idx))
            test_data = pickle.load(open(test_file, 'rb')) if (not args.no_load) and os.path.isfile(test_file) else {}
            print('Verify test_idx =', test_idx)

            net.reset_bounds()
            
            inputs, targets = inputs.to(device), targets.to(device)
            abs_inputs = get_inputs(args.test_domain, inputs, args.test_eps, device, dtype=dtype)
            nat_out = net(inputs)
            nat_ok = targets.eq(nat_out.max(dim=1)[1]).item()
            tot_nat_ok += float(nat_ok)
            test_data['ok'] = nat_ok
            if not nat_ok:
                report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
                continue

            for _ in range(args.attack_restarts):
                with torch.enable_grad():
                    pgd_loss, pgd_ok = get_adv_loss(device, args.test_eps, -1, net, None, inputs, targets, args.test_att_n_steps, args.test_att_step_size)
                    if not pgd_ok:
                        break

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
            if args.no_milp:
                report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
                continue

            zono_inputs = get_inputs('zono_iter', inputs, args.test_eps, device, dtype=dtype)
            bounds = compute_bounds(net, device, len(net.blocks)-1, args, zono_inputs)
            relu_params = reset_params(args, net, dtype)
            with torch.enable_grad():
                learn_slopes(device, relu_params, bounds, args, len(net.blocks), net, inputs, targets, abs_inputs, None, None)
            bounds = compute_bounds(net, device, len(net.blocks)-1, args, zono_inputs)

            for _ in range(args.attack_restarts):
                with torch.enable_grad():
                    latent_loss, latent_ok = get_adv_loss(
                        device, args.test_eps, latent_idx, net, bounds, inputs, targets, args.test_att_n_steps, args.test_att_step_size)
                    # print('-> ', latent_idx, latent_loss, latent_ok)
                    if not latent_ok:
                        break

            if latent_ok:
                tot_attack_ok += 1
            zono_out = net(zono_inputs)
            verified, verified_corr = zono_out.verify(targets)
            test_data['verified'] = int(verified_corr.item())
            if verified_corr:
                tot_verified_corr += 1
                report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
                continue

            loss_after = net(abs_inputs).ce_loss(targets)
            if args.refine_lidx is not None:
                bounds = compute_bounds(net, device, len(net.blocks)-1, args, abs_inputs)
                for lidx in range(0, args.layer_idx+2):
                    net.blocks[lidx].bounds = bounds[lidx]

                print('loss before refine: ', net(abs_inputs).ce_loss(targets))
                refine_dim = bounds[args.refine_lidx+1][0].shape[2]
                pbar = tqdm(total=refine_dim*refine_dim, dynamic_ncols=True)
                for refine_i in range(refine_dim):
                    for refine_j in range(refine_dim):
                        refine(args, bounds, net, refine_i, refine_j, abs_inputs, input_size)
                        pbar.update(1)
                pbar.close()
                loss_after = net(abs_inputs).ce_loss(targets)
                print('loss after refine: ', loss_after)

            if loss_after < args.loss_threshold:
                if args.refine_opt is not None:
                    with torch.enable_grad():
                        learn_bounds(net, bounds, relu_params, zono_inputs, args.refine_opt)
                if verify_test(args, net, inputs, targets, abs_inputs, bounds, test_data, test_idx):
                    tot_verified_corr += 1
                    test_data['verified'] = True
            report(ver_logdir, tot_verified_corr, tot_nat_ok, tot_attack_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
    img_file.close()


if __name__ == '__main__':
    main()
