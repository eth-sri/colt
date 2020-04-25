from gurobipy import GRB, Model, LinExpr
from layers import ReLU, Conv2d


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
            kernel_size, stride, pad = block.kernel_size, block.stride, block.padding
            dim = bounds[lidx+1][0].shape[2]
            out_channels, in_channels = weight.shape[0], weight.shape[1]

            for x in range(0, dim):
                for y in range(0, dim):
                    if (x, y) not in dep[lidx]:
                        continue
                    for out_ch in range(out_channels):
                        expr = LinExpr()
                        expr += bias[out_ch]
                        for kx in range(0, kernel_size):
                            for ky in range(0, kernel_size):
                                new_x = -pad + x * stride + kx
                                new_y = -pad + y * stride + ky
                                if new_x < 0 or new_y < 0 or new_x >= block.bounds[0].size(2) or new_y >= block.bounds[0].size(3):
                                    continue
                                dep[lidx-1][(new_x, new_y)] = True
                                for in_ch in range(in_channels):
                                    if (in_ch, new_x, new_y) not in neurons[lidx-1]:
                                        in_lb = bounds[lidx][0][0, in_ch, new_x, new_y].item()
                                        in_ub = bounds[lidx][1][0, in_ch, new_x, new_y].item()
                                        neurons[lidx-1][(in_ch, new_x, new_y)] = model.addVar(in_lb, in_ub, vtype=GRB.CONTINUOUS)
                                    expr += neurons[lidx-1][(in_ch, new_x, new_y)] * weight[out_ch, in_ch, kx, ky]
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
        # print('new_bounds: ', new_lb, new_ub, ', old bounds: ', old_lb, old_ub)
        if new_lb != -GRB.INFINITY and new_lb >= old_lb:
            net.blocks[args.refine_lidx+1].bounds[0][0, ch_idx, refine_i, refine_j] = new_lb
        if new_ub != GRB.INFINITY and new_ub <= old_ub:
            net.blocks[args.refine_lidx+1].bounds[1][0, ch_idx, refine_i, refine_j] = new_ub
