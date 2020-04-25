import torch
import torch.optim as optim
import time
from layers import ReLU


def learn_slopes(device, relu_params, bounds, args, n_layers, net, inputs, targets, abs_inputs, i, j):
    for param in relu_params:
        param.data = torch.ones(param.size()).to(param.device)
    relu_opt = optim.Adam(relu_params, lr=0.1, weight_decay=0)
    lr_scheduler = optim.lr_scheduler.StepLR(relu_opt, step_size=args.num_iters, gamma=0.1)
    ret_verified = False

    for it in range(args.slope_iters):
        relu_opt.zero_grad()
        if i is None and j is None:
            init_lambda = (it == 0)
            abs_out = net(abs_inputs, init_lambda)
            _, verified_corr = abs_out.verify(targets)
            if verified_corr:
                ret_verified = True
                break
            abs_loss = abs_out.ce_loss(targets)
        else:
            init_lambda = (it == 0)
            abs_out = net(abs_inputs, init_lambda)
            abs_loss = -abs_out.get_min_diff(i, j)
            if abs_loss < 0:
                ret_verified = True
                break

        relu_opt.zero_grad()
        abs_loss.backward()
        relu_opt.step()
        lr_scheduler.step()
        for param in relu_params:
            if param.grad is not None:
                param.data = torch.clamp(param.data, 0, 1)

    relu_priority = {}
    for lidx, layer in enumerate(net.blocks):
        if isinstance(layer, ReLU):
            relu_priority[lidx] = []
            for param in layer.parameters():
                if param.grad is None:
                    continue
                g_abs = param.grad.abs().view(-1)
                for i in range(g_abs.size()[0]):
                    relu_priority[lidx].append(int(g_abs[i]*1000000))

    net.zero_grad()
    return ret_verified, relu_priority


def learn_bounds(net, bounds, relu_params, abs_inputs, refine_opt):
    n_neurons = net.blocks[refine_opt].dims
    new_lb, new_ub = [], []
    t_start = time.time()
    tot_bin = 0
    for neuron_idx in range(n_neurons):
        for is_lb in [True, False]:
            for param in relu_params:
                param.data = torch.ones(param.size()).to(param.device)
            relu_opt = optim.Adam(relu_params, lr=0.1, weight_decay=0)
            lr_scheduler = optim.lr_scheduler.StepLR(relu_opt, step_size=10, gamma=0.5)
            for it in range(50):
                init_lambda = (it == 0)
                abs_curr = abs_inputs
                relu_opt.zero_grad()
                for j, layer in enumerate(net.blocks):
                    lb, ub = abs_curr.concretize()
                    if j == refine_opt:
                        # if is_lb and it == 0:
                        #     print('-> before neuron %d: lb=%.5lf, ub=%.5lf' % (neuron_idx, lb[0, neuron_idx].item(), ub[0, neuron_idx].item()))
                        if is_lb:
                            (-lb[0, neuron_idx]).backward()
                        else:
                            ub[0, neuron_idx].backward()
                        relu_opt.step()
                        for param in relu_params:
                            if param.grad is not None:
                                param.data = torch.clamp(param.data, 0, 1)
                        break
                    if isinstance(layer, ReLU):
                        abs_curr = layer(abs_curr, init_lambda=init_lambda)
                    else:
                        abs_curr = layer(abs_curr)
                if it == 0 and (lb[0, neuron_idx] > 0 or ub[0, neuron_idx] < 0):
                    break
                lr_scheduler.step()
            if is_lb:
                new_lb += [lb[0, neuron_idx].item()]
            else:
                new_ub += [ub[0, neuron_idx].item()]
        # print('neuron %d: old_lb=%.5lf, old_ub=%.5lf' % (
        #     neuron_idx, bounds[refine_opt][0][0, neuron_idx].item(), bounds[refine_opt][1][0, neuron_idx].item()))
        # print('neuron %d: lb=%.5lf, ub=%.5lf' % (neuron_idx, new_lb[-1], new_ub[-1]))
        if new_lb[-1] > bounds[refine_opt][0].data[0, neuron_idx]:
            bounds[refine_opt][0].data[0, neuron_idx] = new_lb[-1]
        if new_ub[-1] < bounds[refine_opt][1].data[0, neuron_idx]:
            bounds[refine_opt][1].data[0, neuron_idx] = new_ub[-1]
        if new_lb[-1] < 0 and new_ub[-1] > 0:
            tot_bin += 1
            # print(neuron_idx, 'tot_bin: ', tot_bin)
    t_end = time.time()
    net.blocks[refine_opt].bounds = bounds[refine_opt]
    print('time to optimize the bounds: ', t_end - t_start)

            
