
import torch
import numpy as np
from data.load_data import load_data
from method import export_method

main_args = None


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(main_args.seed)
    torch.manual_seed(main_args.seed)
    print("seed is {}".format(main_args.seed))

    train_loader, test_loader, num_classes = load_data(main_args)

    method_factory = export_method.__dict__[main_args.method]
    method_params = dict(train_loader=train_loader, eval_loader=test_loader,
                         num_classes=num_classes, args=main_args)
    method = method_factory(**method_params)
    method.train_model()
