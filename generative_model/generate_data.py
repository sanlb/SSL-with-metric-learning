from generative_model.models import DeepGenerativeModel
from generative_model.load_data import get_mnist
import torch
from itertools import cycle
import matplotlib.pyplot as plt
from generative_model.models import SVI, ImportanceWeightedSampler
from torch.autograd import Variable
from generative_model.load_data import onehot
from generative_model.generative_utils import get_logger
import numpy as np
import os
cuda = torch.cuda.is_available()

dataset = 'mnist'
if dataset == 'mnist':
    NUM_TRAIN = 60000
    num_labels = 40
    path_model = './checkpoint/m2mnist-{}.pth'.format(num_labels)
    num_unlabel = NUM_TRAIN - num_labels
elif dataset == 'fashionmnist':
    NUM_TRAIN = 60000
    num_labels = 500
    path_model = './checkpoint/m2fashionmnist-{}.pth'.format(num_labels)
    num_unlabel = NUM_TRAIN - num_labels
else:
    raise NotImplementedError

if not os.path.exists('./checkpoint'):
    os.makedirs('./checkpoint')

data_path = ""
dataroot = data_path + dataset
max_epoch = 10

logger = get_logger('./exp.log')

def one_hot_(label, k):
    y = torch.eye(k)
    label_ = y.index_select(0, label)

    return label_


def main():
    y_dim = 10
    z_dim = 32
    h_dim = [256, 128]
    batch_size = 32
    labels_per_class = num_labels // 10
    model = DeepGenerativeModel([784, y_dim, z_dim, h_dim])
    model.cuda()
    labelled, unlabelled, validation = get_mnist(dataset, location=dataroot, batch_size=batch_size,
                                                 labels_per_class=labels_per_class)
    alpha = 0.1 * len(unlabelled) / len(labelled)

    def binary_cross_entropy(r, x):
        return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
    sampler = ImportanceWeightedSampler(mc=1, iw=1)
    if cuda:
        model = model.cuda()
    elbo = SVI(model, likelihood=binary_cross_entropy, sampler=sampler)

    logger.info('start train......')

    for epoch in range(max_epoch):
        model.train()
        total_loss, accuracy = (0., 0.)
        i = 0
        for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
            # Wrap in variables

            y = one_hot_(y, 10)
            x = x.view(x.size(0), -1).bernoulli()
            u = u.view(u.size(0), -1).bernoulli()
            # x, y, u = Variable(x), Variable(y), Variable(u)

            if cuda:
                # They need to be on the same device and be synchronized.
                x, y = x.cuda(), y.cuda()
                u = u.cuda()

            L = -elbo(x, y)
            U = -elbo(u)

            # Add auxiliary classification loss q(y|x)
            logits = model.classify(x)

            # Regular cross entropy
            classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

            J_alpha = L - alpha * classication_loss + U

            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += J_alpha.item()
            accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

        if epoch % 1 == 0:
            model.eval()
            m = len(unlabelled)
            print("Epoch: {}".format(epoch))
            print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, max_epoch,
                                                                          total_loss / m, accuracy / m))

            total_loss, accuracy = (0, 0)
            for x, y in validation:

                y = one_hot_(y, 10)
                x = x.view(x.size(0), -1).bernoulli()

                if cuda:
                    x, y = x.cuda(), y.cuda()

                # L = -elbo(x, y)
                # U = -elbo(x)

                logits = model.classify(x)
                # classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                # J_alpha = L + alpha * classication_loss + U
                #
                # total_loss += J_alpha.data[0]

                _, pred_idx = torch.max(logits, 1)
                _, lab_idx = torch.max(y, 1)
                accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

            m = len(validation)
            # print("[Validation]\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))

            print("[Validation]\t accuracy: {:.4f}".format(accuracy / m))
            logger.info("[Validation]\t accuracy: {:.4f}".format(accuracy / m))
    torch.save(model, path_model)


main()
seed = 98
np.random.seed(seed)
torch.manual_seed(seed)
model = torch.load(path_model)

model.cpu()
model.eval()
z = Variable(torch.randn(16, 32))

# Generate a batch of 7s
y = Variable(onehot(10)(7).repeat(16, 1))

x_mu = model.sample(z, y)

f, axarr = plt.subplots(1, 16, figsize=(18, 12))

samples = x_mu.data.view(-1, 28, 28).numpy()

for i, ax in enumerate(axarr.flat):
    ax.imshow(samples[i])
    ax.axis("off")

plt.show()