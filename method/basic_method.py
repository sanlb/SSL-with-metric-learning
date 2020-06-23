import re
import os
import scipy.io as sio
from abc import ABCMeta, abstractmethod
from datetime import datetime

import torch
from models import export_models
# from ..utils.log_util import TensorboardLogger
from utils.data_utils import ZCATransformation
from utils.fun_utils import parameters_string
from utils.log_utils import GenericCSV, get_logger


class BasicMethod(metaclass=ABCMeta):
    def __init__(self,
                 train_loader,
                 eval_loader,
                 num_classes,
                 args):


        self.result_folder = "{root}/{runner_name}/{run_idx}/{date:%Y-%m-%d_%H-%M-%S}".format(
            root='../results',
            runner_name=args.method,
            run_idx="{}_lables{}_seed{}".format(args.dataset, args.labels, args.seed),
            date=datetime.now()
        )

        # if args.tflog:
        #     self.tensorboard_logger = TensorboardLogger(self.result_folder)
        # else:
        os.makedirs(self.result_folder, exist_ok=True)
        logger = get_logger(os.path.join(self.result_folder, 'exp.log'))
        pars = vars(args)
        for key, value in pars.items():
            logger.info('{}:{}'.format(key, value))

        self.num_classes = num_classes
        self.args = args
        self.train_loader, self.eval_loader = train_loader, eval_loader
        self.global_step = 0

        self.model = self.create_model(False, args=args)
        print(parameters_string(self.model))

        if args.method == 'mean_teacher':
            self.ema_model = self.create_model(ema=True, args=args)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr, betas=(0.9, 0.999), eps=1e-8)

        if self.args.dataset == 'cifar10':
            mat_contents = sio.loadmat(self.args.ZCAPath)
            transformation_matrix = torch.from_numpy(mat_contents['zca_matrix']).float().cuda()
            transformation_mean = torch.from_numpy(mat_contents['zca_mean'][0]).float().cuda()
            self.zca = ZCATransformation(transformation_matrix, transformation_mean)

        if self.args.resume:
            self._load_checkpoint(self.args.resume)
        else:
            self.best_top1_validate, self.best_top5_validate = None, None
            self.start_epoch = args.start_epoch

    # def log_to_tf(self, name, var, step, train):
    #     if self.args.tflog:
    #         name = "Train/" + name if train else "Test/" + name
    #         self.tensorboard_logger.scalar_summary(name, var, step)
    #     else:
    #         pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def _validate(self, *args):
        pass

    @abstractmethod
    def _save_checkpoint(self, *args):
        pass

    @abstractmethod
    def _load_checkpoint(self, filepath):
        pass

    # @staticmethod
    # @abstractmethod
    # def get_params():
    #     pass

    def create_model(self, ema, args):
        print("=> creating {pretrained} {ema} model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='ema' if ema else '',
            arch=args.arch))

        model_factory = export_models.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=self.num_classes)
        model = model_factory(**model_params)
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

