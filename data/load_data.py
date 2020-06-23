import itertools
import os.path
import torch
from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, SequentialSampler
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, DatasetFolder
from functools import reduce
from operator import __or__
from prefetch_generator import BackgroundGenerator
NO_LABEL = -1
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# datadir = '/home/ann/PycharmProjects/data/cifar10'
# zca_components = np.load(datadir + '/zca_components.npy')
# zca_mean = np.load(datadir + '/zca_mean.npy')

def load_data(args, NO_LABEL=-1):

    path = args.datadir + args.dataset

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif args.dataset == 'mnist':
        mean = (0.5, )
        std = (0.5, )
    elif args.dataset == 'stl10':
        assert False, 'Do not finish stl10 code'
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'svhn':
        train_transform = transforms.Compose([
             transforms.RandomCrop(32, padding=2),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean, std)
        ])
    elif args.dataset == 'mnist':

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = TransformTwice(transforms.Compose([
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=2),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)
        ]))
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(path, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = datasets.CIFAR100(path, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = datasets.SVHN(path, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'mnist':
        train_data = datasets.MNIST(path, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(path, train=False, transform=test_transform, download=True)
        num_classes = 10

    elif args.dataset == 'stl10':
        train_data = datasets.STL10(path, split='train', transform=train_transform, download=True)
        test_data = datasets.STL10(path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)


    labeled_idxs, unlabeled_idxs = spilt_l_u(args.dataset, train_data, args.labels)

    # if args.labeled_batch_size:
    # batch_sampler = TwoStreamBatchSampler(
    #     unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    # else:
    #     assert False, "labeled batch size {}".format(args.labeled_batch_size)

    if args.dataset == 'svhn':
        train_data.labels = np.array(train_data.labels)
        train_data.labels[unlabeled_idxs] = NO_LABEL
    else:
        train_data.targets = np.array(train_data.targets)
        train_data.targets[unlabeled_idxs] = NO_LABEL

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True)

    eval_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader, num_classes

def load_data_val(args):
    path = args.datadir + args.dataset

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif args.dataset == 'mnist':
        mean = (0.5,)
        std = (0.5,)
    elif args.dataset == 'stl10':
        assert False, 'Do not finish stl10 code'
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'svhn':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif args.dataset == 'mnist':

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = TransformTwice(transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]))
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(path, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = datasets.CIFAR100(path, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = datasets.SVHN(path, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'mnist':
        train_data = datasets.MNIST(path, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(path, train=False, transform=test_transform, download=True)
        num_classes = 10

    elif args.dataset == 'stl10':
        train_data = datasets.STL10(path, split='train', transform=train_transform, download=True)
        test_data = datasets.STL10(path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    n_labels = num_classes

    def get_sampler(labels, n=None, n_valid=None):

        if args.dataset == 'mnist':
            labels = train_data.targets.numpy()
        elif args.dataset == 'svhn':
            labels = train_data.labels
        else:
            labels = train_data.targets

        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)

        indices_valid = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
        indices_train = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid + n] for i in range(n_labels)])
        indices_unlabelled = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[n_valid:] for i in range(n_labels)])
        # print (indices_train.shape)
        # print (indices_valid.shape)
        # print (indices_unlabelled.shape)
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled

    # print type(train_data.train_labels)

    labels_per_class = args.num_labels / n_labels
    valid_labels_per_class = args.num_val_labels / n_labels

    train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data,
                                                                   labels_per_class,
                                                                   valid_labels_per_class)

    labelled = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler,
                                           num_workers=args.workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=valid_sampler,
                                             num_workers=args.workers, pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=unlabelled_sampler,
                                             num_workers=args.workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                       pin_memory=True)

    return labelled, validation, unlabelled, test

class TempensDatasetFolder(DatasetFolder):
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        super(TempensDatasetFolder, self).__init__(root, loader, extensions, transform, target_transform, is_valid_file)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class TempensImageFolder(TempensDatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(TempensImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 is_valid_file=is_valid_file)
        self.imgs = self.samples

def create_data_loaders_from_dir(datadir, args):
    # path = '/home/ann/PycharmProjects/data/cifar10'

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=2, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    traindir = os.path.join(datadir, 'by-image/train+val')

    dataset = TempensImageFolder(traindir, train_transform)

    test_data = datasets.CIFAR10(root=datadir, train=False, download=True,
                                 transform=eval_transform)

    with open(args.labels) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines())
    labeled_idxs, unlabeled_idxs = relabel_dataset(dataset, labels)


    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                      shuffle=True)


    eval_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader




def spilt_l_u(dataset, train_data, num_labels, classes=10):

    if dataset == 'cifar10':
        labels = train_data.targets
    elif dataset == 'mnist':
        labels = train_data.targets.numpy()
    elif dataset == 'svhn':
        labels = train_data.labels
    else :
        assert "unknow dataset"
    # v = num_val
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




class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, NO_LABEL
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

class ZCATransformation(object):
    def __init__(self, transformation_matrix, transformation_mean):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        # mat_contents = sio.loadmat(path)
        # transformation_matrix = torch.from_numpy(mat_contents['zca_matrix'])
        # transformation_mean = torch.from_numpy(mat_contents['zca_mean'][0])
        self.transformation_matrix = transformation_matrix
        self.transformation_mean = transformation_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) to be whitened.
        Returns:
            Tensor: Transformed image.
        """
        if tensor.size(1) * tensor.size(2) * tensor.size(3) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor[0].size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        batch = tensor.size(0)

        flat_tensor = tensor.view(batch, -1)
        transformed_tensor = torch.mm(flat_tensor - self.transformation_mean, self.transformation_matrix)

        tensor = transformed_tensor.view(tensor.size())
        return tensor.float()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
        return format_string

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)