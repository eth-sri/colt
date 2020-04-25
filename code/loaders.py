import os
import torch
import torchvision
import torchvision.transforms as transforms


def get_mean_sigma(device, dataset):
    if dataset == 'cifar10':
        mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1))
        sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1))
    else:
        mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
        sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
    return mean.to(device), sigma.to(device)


def get_mnist():
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 28, 1, 10


def get_fashion():
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 28, 1, 10


def get_svhn():
    train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())
    return train_set, test_set, 32, 3, 10


def get_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 32, 3, 10


def get_loaders(args):
    if args.dataset == 'cifar10':
        train_set, test_set, input_size, input_channels, n_class = get_cifar10()
    elif args.dataset == 'mnist':
        train_set, test_set, input_size, input_channels, n_class = get_mnist()
    elif args.dataset == 'fashion':
        train_set, test_set, input_size, input_channels, n_class = get_fashion()
    elif args.dataset == 'svhn':
        train_set, test_set, input_size, input_channels, n_class = get_svhn()
    else:
        raise NotImplementedError('Unknown dataset')

    if args.n_valid is not None:
        print('Using validation set of size %d!' % args.n_valid)
        train_set, test_set = torch.utils.data.random_split(train_set, [len(train_set) - args.n_valid, args.n_valid])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch, shuffle=True, num_workers=8, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False, num_workers=8, drop_last=True)
    return len(train_set), train_loader, test_loader, input_size, input_channels, n_class

