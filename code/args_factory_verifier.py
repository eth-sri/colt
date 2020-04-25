import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Perform greedy layerwise training.')
    parser.add_argument('--dataset', default='cifar10', help='dataset to use')
    parser.add_argument('--net', required=True, type=str, help='network to use')
    parser.add_argument('--test_eps', required=True, type=float, help='epsilon to verify')
    parser.add_argument('--load_model', type=str, help='model to load')
    parser.add_argument('--test_domain', default='zono_iter', type=str, help='domain to test with')
    parser.add_argument('--num_iters', default=100, type=int, help='number of iterations to find slopes')

    parser.add_argument('--test_att_n_steps', default=None, type=int, help='number of steps for the attack')
    parser.add_argument('--test_att_step_size', default=None, type=float, help='step size for the attack (relative to epsilon)')
    parser.add_argument('--attack_restarts', default=20, type=int, help='number of restarts for the attack')

    parser.add_argument('--latent_idx', default=None, type=int, help='layer index where to perform latent attack')
    parser.add_argument('--layer_idx', default=1, type=int, help='layer index of flattened vector')
    parser.add_argument('--n_valid', default=None, type=int, help='number of test samples')
    parser.add_argument('--n_train', default=None, type=int, help='number of training samples to use')
    parser.add_argument('--train_batch', default=1, type=int, help='batch size for training')
    parser.add_argument('--test_batch', default=128, type=int, help='batch size for testing')

    parser.add_argument('--unverified_imgs_file', type=str, default='unverified_imgs.csv', help='save images that were not verified')
    parser.add_argument('--fail_break', action='store_true', help='break if one class fails')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--no_milp', action='store_true', help='no MILP mode')
    parser.add_argument('--no_load', action='store_true', help='verify from scratch')
    parser.add_argument('--no_smart', action='store_true', help='bla')
    parser.add_argument('--milp_timeout', default=1000, type=int, help='timeout for MILP')
    parser.add_argument('--eval_train', action='store_true', help='evaluate on training set')
    parser.add_argument('--start_idx', default=0, type=int, help='specific index to start')
    parser.add_argument('--end_idx', default=1000, type=int, help='specific index to end')
    parser.add_argument('--max_binary', default=None, type=int, help='number of neurons to encode as binary variable in MILP (per layer)')
    parser.add_argument('--tot_binary', default=None, type=int, help='number of neurons to encode as binary variable in MILP (total)')
    parser.add_argument('--refine_lidx', default=None, type=int, help='layer to refine')
    parser.add_argument('--refine_milp', default=0, type=int, help='number of neurons to refine using MILP')
    parser.add_argument('--refine_opt', default=None, type=int, help='index of layer to refine via optimization')
    parser.add_argument('--slope_iters', default=500, type=int, help='number of iterations to learn the slopes')
    parser.add_argument('--loss_threshold', default=1e9, type=float, help='threshold to consider for MILP verification')

    parser.add_argument('--n-rand-proj', default=50, type=int, help='number of random projections')
    parser.add_argument('--layers', required=False, default=None, type=int, nargs='+', help='layer indices for training')
    return parser.parse_args()
