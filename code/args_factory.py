import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Convex layerwise adversarial training.')
    
    # Basic arguments
    parser.add_argument('--train-mode', default='train', type=str, help='whether to train adversarially')
    parser.add_argument('--dataset', default='cifar10', help='dataset to use')
    parser.add_argument('--net', required=True, type=str, help='network to use')
    parser.add_argument('--train-batch', default=100, type=int, help='batch size for training')
    parser.add_argument('--test-batch', default=100, type=int, help='batch size for testing')
    parser.add_argument('--layers', required=False, default=None, type=int, nargs='+', help='layer indices for training')
    parser.add_argument('--n-epochs', default=1, type=int, help='number of epochs')
    parser.add_argument('--mix-epochs', default=1, type=int, help='number of epochs to anneal schedule')
    parser.add_argument('--n-epochs-reduce', default=0, type=int, help='number of epochs to reduce each layer')
    parser.add_argument('--load-model', default=None, type=str, help='model to load')
    parser.add_argument('--n-valid', default=None, type=int, help='number of validation samples (none to use no validation)')
    parser.add_argument('--test-freq', default=50, type=int, help='frequency of testing')

    # Optimizer and learning rate scheduling
    parser.add_argument('--opt', default='adam', type=str, help='optimizer to use')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr-sched', default='step_lr', type=str, choices=['step_lr', 'cycle'], help='choice of learning rate scheduling')
    parser.add_argument('--lr-step', default=10, type=int, help='number of epochs between lr updates')
    parser.add_argument('--lr-factor', default=0.5, type=float, help='factor by which to decrease learning rate')
    parser.add_argument('--lr-layer-dec', default=0.5, type=float, help='factor by which to decrease learning rate in the next layers')
    parser.add_argument('--pct_start', default=0.3, type=float, help='how long to increase learning rate')

    # Losses and regularizers
    parser.add_argument('--nat-factor', default=0.0, type=float, help='factor for natural loss')
    parser.add_argument('--relu-stable', required=False, type=float, default=None, help='factor for relu stability')
    parser.add_argument('--relu-stable-factor', required=False, type=float, default=1.0, help='factor for relu stability')
    parser.add_argument('--l1-reg', default=0.0, type=float, help='l1 regularization coefficient')
    parser.add_argument('--mix', action='store_true', help='whether to mix adversarial and standard loss')

    # Configuration of adversarial attacks
    parser.add_argument('--train-eps', default=None, required=True, type=float, help='epsilon to train with')
    parser.add_argument('--test-eps', default=None, type=float, help='epsilon to verify')
    parser.add_argument('--anneal', action='store_true', help='whether to anneal epsilon')
    parser.add_argument('--eps-factor', default=1.05, type=float, help='factor to increase epsilon per layer')
    parser.add_argument('--start-eps-factor', default=1.0, type=float, help='factor to determine starting epsilon')
    parser.add_argument('--train-att-n-steps', default=10, type=int, help='number of steps for the attack')
    parser.add_argument('--train-att-step-size', default=0.25, type=float, help='step size for the attack (relative to epsilon)')
    parser.add_argument('--test-att-n-steps', default=None, type=int, help='number of steps for the attack')
    parser.add_argument('--test-att-step-size', default=None, type=float, help='step size for the attack (relative to epsilon)')
    parser.add_argument('--n-rand-proj', default=50, type=int, help='number of random projections')
    parser.add_argument('--train-domain', default=None, type=str, help='domain to train with')
    parser.add_argument('--test-domains', default=[], type=str, nargs='+', help='domains to test with')

    # Metadata
    parser.add_argument('--exp-name', default='dev', type=str, help='name of the experiment')
    parser.add_argument('--exp-id', default=1, type=int, help='name of the experiment')
    parser.add_argument('--no-cuda', action='store_true', help='whether to use only cpu')
    parser.add_argument('--root-dir', required=False, default='./', type=str, help='directory to store the data')
    parser.add_argument('--sigopt-token', default='dev', type=str, choices=['api', 'dev'], help='which sigopt token to use')
    parser.add_argument('--sigopt-exp-id', default=None, type=int, help='id of the sigopt experiment')

    args = parser.parse_args()

    if args.test_eps is None:
        args.test_eps = args.train_eps
    if args.test_att_n_steps is None:
        args.test_att_n_steps = args.train_att_n_steps
        args.test_att_step_size = args.train_att_step_size

    return args
