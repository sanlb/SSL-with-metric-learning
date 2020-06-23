import math
import sys
import os
import torch
import torch.nn as nn

class GaussianNoise(nn.Module):

    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std
    def forward(self, x):
        shape = torch.size(x)
        noise = torch.zeros(shape).cuda()
        noise.data.normal_(0, std=self.std)
        return x + noise

def rampup(epoch, rampup_epoch):
    if epoch < rampup_epoch:
        p = max(0.0, float(epoch)) / float(rampup_epoch)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0


def rampdown(epoch, epochs, rampdown_epoch):
    if epoch >= (epochs - rampdown_epoch):
        ep = (epoch - (epochs -rampdown_epoch)) * 0.5
        return math.exp(-(ep * ep) / rampdown_epoch)
    else:
        return 1.0


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn

class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count:
            self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(-1).sum().float(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
    return res


def save_checkpoint_to_file(state, epoch, is_best, ckptfolder):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(ckptfolder, filename)
    try:
        torch.save(state, checkpoint_path)
    except:
        os.makedirs(ckptfolder)
        torch.save(state, checkpoint_path)
    print("--- checkpoint saved to %s ---" % checkpoint_path)


def save_best_checkpoint_to_file(state, is_best, ckptfolder):
    if is_best:
        best_path = os.path.join(ckptfolder, 'best.ckpt')
        try:
            torch.save(state, best_path)
        except:
            os.makedirs(ckptfolder)
            torch.save(state, best_path)
        print("--- best checkpoint saved to %s ---" % best_path)


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


