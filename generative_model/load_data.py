import torch
import numpy as np
import sys
import torchvision.transforms as transforms
from torchvision import datasets
from functools import reduce
import torch.nn as nn
from operator import __or__
sys.path.append("../semi-supervised")

torch.manual_seed(1337)
np.random.seed(1337)

cuda = torch.cuda.is_available()

def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode



def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

n_labels = 10
def get_mnist(dataset, location="./", batch_size=32, labels_per_class=2):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST, FashionMNIST
    import torchvision.transforms as transforms

    flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1).bernoulli()

    if dataset == 'mnist':
        mnist_train = MNIST(location, train=True, download=True,
                        transform=transforms.ToTensor())
        mnist_test = MNIST(location, train=False, download=True,
                        transform=transforms.ToTensor())
    elif dataset == 'fashionmnist':
        mnist_train = FashionMNIST(location, train=True, download=True,
                            transform=transforms.ToTensor())
        mnist_test = FashionMNIST(location, train=False, download=True,
                           transform=transforms.ToTensor())

    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, pin_memory=cuda,
                                           sampler=get_sampler(mnist_train.targets.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, pin_memory=cuda,
                                             sampler=get_sampler(mnist_train.targets.numpy()))
    test = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, pin_memory=cuda,
                                             sampler=get_sampler(mnist_test.targets.numpy()))

    return labelled, unlabelled, test

def spilt_l_u(train_data, num_labels, num_val=400, classes=10):

    labels = train_data.targets.numpy()

    v = num_val
    n = int(num_labels / classes)
    (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(classes)]))
    # Ensure uniform distribution of labels
    np.random.shuffle(indices)

    indices_train = np.hstack(
        [list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(classes)])
    indices_unlabelled = np.hstack(
        [list(filter(lambda idx: labels[idx] == i, indices))[n:] for i in range(classes)])

    indices_train = torch.from_numpy(indices_train)
    indices_unlabelled = torch.from_numpy(indices_unlabelled)


    return indices_train, indices_unlabelled

def load_data(path, dataset, num_labels, batch_size, workers=4):

    if dataset == 'mnist':
        train_transform = (transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]))

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_data = datasets.MNIST(root=path, train=True, download=True,
                                      transform=train_transform)

        test_data = datasets.MNIST(root=path, train=False, download=True,
                                     transform=eval_transform)
    elif dataset == 'FashionMNIST':
        train_transform = (transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]))

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_data = datasets.FashionMNIST(root=path, train=True, download=True,
                                    transform=train_transform)

        test_data = datasets.FashionMNIST(root=path, train=False, download=True,
                                   transform=eval_transform)
    else:
        assert False, "Unknow dataset"
    labeled_idxs, unlabeled_idxs = spilt_l_u(train_data, num_labels)

    train_data.targets = np.array(train_data.targets)
    train_data.targets[unlabeled_idxs] = -1

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader

def add_fake_data(path_model, generate_n, classes=10):
    print(path_model)
    generate_model = torch.load(path_model)
    generate_model.cpu()
    generate_model.eval()

    per_n = generate_n // classes
    fake_data = []
    fake_target = []
    for i in range(10):
        z = torch.randn(per_n, 32)
        y = onehot(classes)(i).repeat(per_n, 1)
        x_mu = generate_model.sample(z, y)
        fake_x = x_mu.data.view(-1, 1, 28, 28)
        fake_data.append(fake_x)
        fake_target.append(torch.argmax(y, dim=1))

    fake_data = torch.cat(fake_data, dim=0)
    fake_target = torch.cat(fake_target, dim=0)
    return fake_data, fake_target

class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def _get_pos_neg_mask(self, labels):
        indices_equal = np.cast[np.bool](np.eye(np.shape(labels)[0], dtype=np.int32))
        indices_not_equal = np.logical_not(indices_equal)
        pos_mask = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))
        neg_mask = ~pos_mask
        pos_mask = np.logical_and(pos_mask, indices_not_equal)

        return pos_mask, neg_mask

    def _ml_loss_pos(self, sim_mat, mask):
        exp_ = torch.exp(-self.scale_pos * (sim_mat - self.thresh))
        loss = torch.sum(torch.mul(exp_, mask), 1)
        loss = 1.0 / self.scale_pos * torch.log(1 + loss)
        return loss

    def _ml_loss_neg(self, sim_mat, mask):
        exp_ = torch.exp(self.scale_neg * (sim_mat - self.thresh))
        loss = torch.sum(torch.mul(exp_, mask), 1)
        loss = 1.0 / self.scale_neg * torch.log(1 + loss)
        return loss

    def _loss_mask(self, feats, labels, probability_v):
        probability_v = probability_v.view(-1, 1)
        pos_mask, neg_mask = self._get_pos_neg_mask(labels)
        sim_mat = torch.matmul(feats, torch.t(feats))
        probability_m = torch.matmul(probability_v, torch.t(probability_v))

        probability_m = torch.where(probability_m > 0.3, probability_m, torch.zeros_like(probability_m))
        probability_m = torch.where(probability_m < 0.8, probability_m, torch.zeros_like(probability_m))
        # probability_m = (1. - probability_m)

        mask_neg = torch.Tensor(np.cast[np.float](neg_mask)).cuda()
        mask_pos = torch.Tensor(np.cast[np.float](pos_mask)).cuda()
        neg_pair = torch.mul(mask_neg, sim_mat)
        pos_pair = torch.mul(mask_pos, sim_mat)

        ones = torch.ones_like(pos_pair).cuda()
        anchor_negative_dist = pos_pair + torch.mul(ones, (1.0 - mask_pos))

        pos_min, _ = torch.min(anchor_negative_dist, 1)
        neg_max, _ = torch.max(neg_pair, 1)
        mask_n = (sim_mat + self.margin > pos_min.view(-1, 1)).float().cuda()
        mask_p = (sim_mat - self.margin < neg_max.view(-1, 1)).float().cuda()

        mask_neg = torch.mul(mask_neg, mask_n)
        mask_pos = torch.mul(mask_pos, mask_p)

        w_mask_neg = torch.mul(mask_neg, probability_m)
        w_mask_pos = torch.mul(mask_pos, probability_m)

        pos_loss = self._ml_loss_pos(sim_mat, w_mask_pos)
        neg_loss = self._ml_loss_neg(sim_mat, w_mask_neg)

        loss = torch.mean(pos_loss + neg_loss)

        return loss

    def forward(self, feats, labels, probability_v):

        loss = self._loss_mask(feats, labels, probability_v)

        return loss