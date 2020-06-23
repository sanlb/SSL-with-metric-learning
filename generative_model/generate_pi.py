import torch
import torch.optim as optim
import torch.nn as nn
from generative_model.models import CNN
import numpy as np
from generative_model.load_data import load_data, add_fake_data, MultiSimilarityLoss
import torch.nn.functional as F
from torch.autograd import Variable

dataset = 'mnist'#FashionMNIST mnist
if dataset == 'mnist':
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    num_labeled = 100
    generate_n = 0
    path_model = ''
elif dataset == 'FashionMNIST':
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    num_labeled = 500
    generate_n = 0
    path_model = ''
else:
    raise NotImplementedError

batch_size = 32
eval_batch_size = 100
num_iter_per_epoch = int((NUM_TRAIN + generate_n) / eval_batch_size)
eval_freq = 1
max_learning_rate = 0.001
use_cuda = True
top_bn = True
epoch_decay_start = 80
num_epochs = 120

labels_per_class = num_labeled // 10
path = "D:/code/python/data/"
dataroot = path + dataset


def masked_crossentropy(out, labels):
    cond = (labels >= 0)
    nnz = torch.nonzero(cond)
    nbsup = len(nnz)
    # check if labeled samples in batch, return 0 if none
    if nbsup > 0:
        masked_outputs = torch.index_select(out, 0, nnz.view(nbsup))
        masked_labels = labels[cond]
        loss = F.cross_entropy(masked_outputs, masked_labels)
        # ce_loss = ce(masked_outputs, masked_labels)
        return loss
    return Variable(torch.FloatTensor([0.]).cuda(), requires_grad=True)

def add_noise2x(x, input_shape=(1, 28, 28)):
    shape = (x.shape[0],) + input_shape
    noise = torch.zeros(shape)
    noise.data.normal_(0, std=0.15)
    return x + noise

def tocuda(x):
    if use_cuda:
        return x.cuda()
    return x

def softmax_mse_loss(input_logits, target_logits):

    assert input_logits.size() == target_logits.size(), print(input_logits.size(), target_logits.size())
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes

def train(model, x1, x2, y, optimizer):
    # feature_cri = MultiSimilarityLoss()

    min_batch = len(y)

    y_pred1 = model(x1)
    y_pred2 = model(x2)

    ce_loss = masked_crossentropy(y_pred1, y)

    mse = softmax_mse_loss(y_pred1, y_pred2) / min_batch
    # y = y.cpu()
    # msl = feature_cri(h1, y)

    loss = ce_loss + mse

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, ce_loss

def eval_accuarcy(model, x, y):

    y_pred = model(x)
    prob, idx = torch.max(y_pred, dim=1)
    return torch.eq(idx, y).float().mean()

def result():

    train_loader, test_loader = load_data(path=dataroot, dataset=dataset,
                                          num_labels=num_labeled, batch_size=batch_size)

    train_data = []
    train_target = []

    for i, (data, target) in enumerate(train_loader):
        train_data.append(data)
        train_target.append(target)

    train_data = torch.cat(train_data, dim=0)
    train_target = torch.cat(train_target, dim=0)
    if generate_n > 0:
        fake_data, fake_target = add_fake_data(path_model, generate_n)
        train_data = torch.cat([train_data, fake_data], dim=0)
        train_target = torch.cat([train_target, fake_target], dim=0)

    idx = torch.where(train_target > -1)
    print(len(idx))

    rd = np.random.permutation(np.arange(train_data.size(0)))
    train_data = train_data[rd]
    train_target = train_target[rd]

    model = tocuda(CNN())
    optimizer = optim.Adam(model.parameters(), lr=max_learning_rate)

    # train the network
    for epoch in range(num_epochs):

        if epoch > epoch_decay_start:
            decayed_lr = (num_epochs - epoch) * max_learning_rate / (num_epochs - epoch_decay_start)
            optimizer.lr = decayed_lr
            optimizer.betas = (0.5, 0.999)

        for i in range(num_iter_per_epoch):

            batch_indices = torch.LongTensor(np.random.choice(train_data.size()[0],
                                                              eval_batch_size, replace=False))
            x = train_data[batch_indices]
            y = train_target[batch_indices]
            x1 = add_noise2x(x)
            x2 = add_noise2x(x)
            loss, ce_loss = train(model.train(), tocuda(x1), tocuda(x2), tocuda(y), optimizer)

        # if epoch % eval_freq == 0 or epoch + 1 == num_epochs:
        #     test_accuracy = 0.0
        #     counter = 0.0
        #     for (data, target) in test_loader:
        #         n = data.size()[0]
        #         acc = eval_accuarcy(model.eval(), tocuda(data), tocuda(target))
        #         test_accuracy += n * acc
        #         counter += n
        #     test_acc = test_accuracy.item() / counter
        #     print("Epoch : %.d, Full test accuracy : %.4f" % (epoch, test_accuracy.item() / counter))

    # np.save('./pi_acc.npy', results)
    # torch.save(model, path_model)

    test_accuracy = 0.0
    counter = 0

    for (data, target) in test_loader:
        n = data.size()[0]
        acc = eval_accuarcy(model.eval(), tocuda(data), tocuda(target))
        test_accuracy += n*acc
        counter += n
    best_acc = float(test_accuracy.item()) / counter
    print("Full test accuracy :", best_acc)

    return best_acc

def eval_target(model_path='./checkpoint/m2mnist-1000.pth'):
    train_loader, test_loader = load_data(path=dataroot, dataset=dataset,
                                          num_labels=num_labeled, batch_size=batch_size)

    model = torch.load(model_path)
    model.cuda()
    model.eval()

    for i, (data, target) in enumerate(test_loader):

        y_pred = model(data.cuda())

        if i == 1:
            print('target', target)
            y_ = F.softmax(y_pred, dim=1)
            print(y_[:30])
            prob, idx = torch.max(y_, dim=1)
            print(prob[:30])
            print(idx[:30])
            break


##### num_label = 1000

if __name__ == '__main__':

    # eval_target()

    run_num = 1
    results = np.zeros((run_num, 1))
    results_temp = np.zeros((run_num, 1))
    temp_l = [20, 30, 40, 100]
    temp_n = [400, 800]
    for i in range(len(temp_l)):
        for j in range(len(temp_n)):
            num_labeled = temp_l[i]
            generate_n = temp_n[j]
            path_model = './checkpoint/m2mnist-{}.pth'.format(num_labeled)
            for a in range(run_num):
                results_temp[a] = result()

            print('num:{}, g_num:{}'.format(num_labeled, generate_n))
            print(results_temp)
            print(np.mean(results_temp, axis=0))
