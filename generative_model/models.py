
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from itertools import repeat
from .vae import VariationalAutoencoder, Encoder, Decoder
from generative_model.generative_utils import *
from torch.nn.utils import weight_norm

class GaussianNoise(nn.Module):

    def __init__(self, batch_size=100, input_shape=(1, 28, 28), std=0.15):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.std = std

    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        return x + self.noise

class CNN(nn.Module):

    def __init__(self, channel=1, std=0.15, p=0.5, fm1=16, fm2=32):
        super(CNN, self).__init__()
        self.channel = channel
        self.fm1   = fm1
        self.fm2   = fm2
        self.std   = std
        # self.gn    = GaussianNoise(batch_size, std=self.std)
        self.act   = nn.LeakyReLU(0.1)
        self.drop  = nn.Dropout(p)
        self.conv1 = weight_norm(nn.Conv2d(self.channel, self.fm1, 3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.mp    = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc    = nn.Linear(self.fm2 * 7 * 7, 10)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        x = self.act(self.mp(self.conv1(x)))
        x = self.act(self.mp(self.conv2(x)))
        x = x.view(-1, self.fm2 * 7 * 7)
        x = self.drop(x)
        x = self.fc(x)
        # x = self.softmax(x)
        return x

class CNN_ml(nn.Module):

    def __init__(self, channel=1, std=0.15, p=0.5, fm1=16, fm2=32):
        super(CNN_ml, self).__init__()
        self.channel = channel
        self.fm1   = fm1
        self.fm2   = fm2
        self.std   = std
        # self.gn    = GaussianNoise(batch_size, std=self.std)
        self.act   = nn.LeakyReLU(0.1)
        self.drop  = nn.Dropout(p)
        self.conv1 = (nn.Conv2d(self.channel, self.fm1, 3, padding=1))
        self.conv2 = (nn.Conv2d(self.fm1, self.fm2, 3, padding=1))
        self.conv3 = nn.Conv2d(self.fm2, self.fm2, 3, padding=1)
        self.mp    = nn.MaxPool2d(3, stride=2, padding=1)
        self.ap = nn.AvgPool2d(7, 7)
        self.fc    = nn.Linear(self.fm2, 10)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        x = self.mp(self.act(self.conv1(x)))
        x = self.drop(x)
        x = self.mp(self.act(self.conv2(x)))
        x = self.drop(x)
        x = self.act(self.conv3(x))
        x = self.ap(x)
        x = x.view(-1, self.fm2)
        h = nn.functional.normalize(x, p=2, dim=1)
        x = self.fc(h)
        # x = self.softmax(x)
        return x, h


class Classifier(nn.Module):
    def __init__(self, dims):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims
        self.dense = nn.Linear(x_dim, h_dim)
        self.logits = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.dense(x))
        x = F.softmax(self.logits(x), dim=-1)
        return x


class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(self, dims):
        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder([x_dim + self.y_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
        self.classifier = Classifier([x_dim, h_dim[0], self.y_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))

        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        x_mu = self.decoder(torch.cat([z, y], dim=1))

        return x_mu

    def classify(self, x):
        logits = self.classifier(x)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x


class ImportanceWeightedSampler(object):
    """
    Importance weighted sampler [Burda 2015] to
    be used in conjunction with SVI.
    """
    def __init__(self, mc=1, iw=1):
        """
        Initialise a new sampler.
        :param mc: number of Monte Carlo samples
        :param iw: number of Importance Weighted samples
        """
        self.mc = mc
        self.iw = iw

    def resample(self, x):
        return x.repeat(self.mc * self.iw, 1)

    def __call__(self, elbo):
        elbo = elbo.view(self.mc, self.iw, -1)
        elbo = torch.mean(log_sum_exp(elbo, dim=1, sum_op=torch.mean), dim=0)
        return elbo.view(-1)


class DeterministicWarmup(object):
    """
    Linear deterministic warm-up as described in
    [Sønderby 2016].
    """
    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1/n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc

        self.t = self.t_max if t > self.t_max else t
        return self.t


class SVI(nn.Module):
    """
    Stochastic variational inference (SVI).
    """
    base_sampler = ImportanceWeightedSampler(mc=1, iw=1)
    def __init__(self, model, likelihood=F.binary_cross_entropy, beta=repeat(1), sampler=base_sampler):
        """
        Initialises a new SVI optimizer for semi-
        supervised learning.
        :param model: semi-supervised model to evaluate
        :param likelihood: p(x|y,z) for example BCE or MSE
        :param sampler: sampler for x and y, e.g. for Monte Carlo
        :param beta: warm-up/scaling of KL-term
        """
        super(SVI, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.sampler = sampler
        self.beta = beta

    def forward(self, x, y=None):
        is_labelled = False if y is None else True

        # Prepare for sampling
        xs, ys = (x, y)

        # Enumerate choices of label
        if not is_labelled:
            ys = enumerate_discrete(xs, self.model.y_dim)
            xs = xs.repeat(self.model.y_dim, 1)

        # Increase sampling dimension
        # xs = self.sampler.resample(xs)
        # ys = self.sampler.resample(ys)

        reconstruction = self.model(xs, ys)

        # p(x|y,z)
        likelihood = -self.likelihood(reconstruction, xs)

        # p(y)
        prior = -log_standard_categorical(ys)

        # Equivalent to -L(x, y)
        elbo = likelihood + prior - next(self.beta) * self.model.kl_divergence
        L = self.sampler(elbo)

        if is_labelled:
            return torch.mean(L)

        logits = self.model.classify(x)

        L = L.view_as(logits.t()).t()

        # Calculate entropy H(q(y|x)) and sum over all labels
        H = -torch.sum(torch.mul(logits, torch.log(logits + 1e-8)), dim=-1)
        L = torch.sum(torch.mul(logits, L), dim=-1)

        # Equivalent to -U(x)
        U = L + H
        return torch.mean(U)