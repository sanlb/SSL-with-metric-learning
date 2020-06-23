
import numpy as np
from .basic_method import BasicMethod
import torch
from utils.loss_utils import softmax_mse_loss
from utils.fun_utils import rampdown, rampup, AverageMeter, AverageMeterSet
from utils.fun_utils import accuracy, save_best_checkpoint_to_file, save_checkpoint_to_file
import time
import os


class TemporalEnsembling(BasicMethod):

    def __init__(self,
                 train_loader,
                 eval_loader,
                 num_classes,
                 args):
        super(TemporalEnsembling, self).__init__(train_loader, eval_loader, num_classes, args)
        self.loss_ce = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-1).cuda()
        self.loss_pi = softmax_mse_loss
        self.num_classes = num_classes

    def adjust_optimizer_params(self, optimizer, epoch):
        rampup_value = rampup(epoch, self.args.rampup_epoch)
        rampdown_value = rampdown(epoch, self.args.epochs, self.args.rampdown_epoch)
        learning_rate = rampup_value * rampdown_value * self.args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
            old_beta1, old_beta2 = param_group['betas']
            adam_beta1 = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
            param_group['betas'] = adam_beta1, old_beta2
        return learning_rate

    def adjust_consistency_weight(self, epoch):
        return self.args.consistency * rampup(epoch, self.args.rampup_epoch)

    def train_model(self):
        train_data = []
        train_target = []
        lengths = len(self.train_loader)
        for i, (data, target) in enumerate(self.train_loader):
            train_data.append(data)
            train_target.append(target)

        train_data = torch.cat(train_data, dim=0)
        train_target = torch.cat(train_target, dim=0)

        Z = torch.zeros(lengths, self.num_classes).float().cuda()  # intermediate values
        z = torch.zeros(lengths, self.num_classes).float().cuda()  # temporal outputs
        outputs = torch.zeros(lengths, self.num_classes).float().cuda()  # current outputs

        for epoch in range(self.start_epoch, self.args.epochs):
            print("epoch {e}".format(e=epoch))
            start = time.time()
            if epoch == self.start_epoch:
                cur_cons_weight = 0
            else:
                cur_cons_weight = self.adjust_consistency_weight(epoch)

            Z, z, outputs = self._train_one_epoch(train_data, train_target, epoch, cur_cons_weight, Z, z, outputs)

            Z = self.args.alpha * Z + (1. - self.args.alpha) * outputs
            z = Z * (1. / (1. - self.args.alpha ** (epoch + 1)))

            top1_avg_validate, top5_avg_validate, class_loss_avg_validate = self._validate(epoch)

            if self.best_top1_validate is None or self.best_top1_validate < top1_avg_validate:
                self.best_top1_validate = top1_avg_validate
                self.best_top5_validate = top5_avg_validate
                is_best = True
            else:
                is_best = False

            self._save_checkpoint(epoch, self.global_step, top1_avg_validate, top5_avg_validate,
                                  self.best_top1_validate, self.best_top5_validate,
                                  class_loss_avg_validate, is_best)

            # self.training_csv.add_data(*list(self.map.values()))

            end = time.time()
            print("EPOCH {e} use {time} s".format(e=epoch, time=(end - start)))
        print("best test top1 accuracy is {acc}".format(acc=self.best_top1_validate))
        # self.training_csv.close()

    def _train_one_epoch(self, train_data, train_target, epoch, cons_weight, Z, z, outputs):

        batch_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_pi = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses_all = AverageMeter()

        start = time.time()

        self.adjust_optimizer_params(self.optimizer, epoch)

        self.model.train()
        total_data_size, total_labeled_size = 0, 0
        num_iter_per_epoch = len(self.train_loader) // self.args.batch_size

        for i in range(num_iter_per_epoch):

            batch_indices = np.random.choice(train_data.size(0), self.args.batch_size, replace=False).long()
            input_pack = train_data[batch_indices]
            target = train_target[batch_indices]

            zcomp = z[batch_indices]
            target_var = target.long().cuda()
            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(-1).sum().float()
            total_data_size += minibatch_size
            total_labeled_size += labeled_minibatch_size

            if len(input_pack) == 2:
                (input1, input2) = input_pack
                input1 = input1.cuda()
                if self.args.dataset == 'cifar10':
                    input1 = self.zca(input1)
                output_1 = self.model(input1)

            else:
                input1 = input_pack.cuda()
                if self.args.dataset == 'cifar10':
                    input1 = self.zca(input1)
                output_1 = self.model(input1)

            loss_ce = self.loss_ce(output_1, target_var) / minibatch_size
            loss_pi = cons_weight * self.loss_pi(output_1, zcomp) / minibatch_size

            loss = loss_ce + loss_pi
            losses_all.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            outputs[batch_indices] = output_1.item()

            if self.args.topk == 1:
                prec1 = accuracy(output_1.data, target_var.data, topk=(1,))[0]
            else:
                prec1, prec5 = accuracy(output_1.data, target_var.data, topk=(1, 5))
                top5.update(prec5.item(), labeled_minibatch_size)

            losses_ce.update(loss_ce.item())
            losses_pi.update(loss_pi.item())
            top1.update(prec1.item(), labeled_minibatch_size)

            batch_time.update(time.time() - start)
            self.global_step += 1

        return Z, z, outputs

    def _validate(self, epoch=None):
        class_criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1).cuda()
        meters = AverageMeterSet()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()

        with torch.no_grad():
            for i, (input, target) in enumerate(self.eval_loader):
                meters.update('data_time', time.time() - end)
                input = input.cuda()
                target = target.long().cuda()
                if self.args.dataset == 'cifar10':
                    input = self.zca(input)

                minibatch_size = len(target)
                labeled_minibatch_size = target.data.ne(-1).sum().float()
                assert labeled_minibatch_size and minibatch_size == labeled_minibatch_size
                meters.update('labeled_minibatch_size', labeled_minibatch_size.item())

                # compute output
                output = self.model(input)
                class_loss = class_criterion(output, target) / minibatch_size

                # measure accuracy and record loss
                if self.args.topk == 1:
                    prec1 = accuracy(output.data, target.data, topk=(1,))[0]
                    meters.update('top5', 0, labeled_minibatch_size.item())
                    meters.update('error5', 100.0, labeled_minibatch_size.item())
                else:
                    prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
                    meters.update('top5', prec5[0], labeled_minibatch_size.item())
                    meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size.item())
                meters.update('class_loss', class_loss.item(), labeled_minibatch_size.item())
                meters.update('top1', prec1[0], labeled_minibatch_size.item())
                meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size.item())

                # measure elapsed time
                meters.update('batch_time', time.time() - end)
                end = time.time()

            print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\tClass_Loss {cl.avg:.5f}'
                  .format(top1=meters['top1'], top5=meters['top5'], cl=meters['class_loss']))

        return meters['top1'].avg, meters['top5'].avg, meters['class_loss'].avg

    def _update_ema_variables(self, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _save_checkpoint(self,
                         epoch, global_step, top1_validate, top5_validate,
                         best_top1_validate, best_top5_validate,
                         class_loss_validate, is_best):
        if not self.args.tflog:
            return

        if is_best:
            save_best_checkpoint_to_file({
                'epoch': epoch,
                'global_step': global_step,
                'semi-supervised-method': self.args.method,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'top1_validate': top1_validate,
                'top5_validate': top5_validate,
                'best_top1_validate': best_top1_validate,
                'best_top5_validate': best_top5_validate,
                'class_loss_validate': class_loss_validate
            }, is_best, self.result_folder)

        if epoch % self.args.checkpoint_epochs == 0:
            save_checkpoint_to_file({
                'epoch': epoch,
                'global_step': global_step,
                'semi-supervised-method': self.args.method,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'top1_validate': top1_validate,
                'top5_validate': top5_validate,
                'best_top1_validate': best_top1_validate,
                'best_top5_validate': best_top5_validate,
                'class_loss_validate': class_loss_validate
            }, epoch, is_best, self.result_folder)

    def _load_checkpoint(self, filepath):
        if not self.args.tflog:
            return

        if os.path.isfile(filepath):
            print("=> loading checkpoint '{}'".format(filepath))
            checkpoint = torch.load(filepath)
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step'] + 1
            self.best_top1_validate = checkpoint['best_top1_validate']
            self.best_top5_validate = checkpoint['best_top5_validate']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) "
                  "best_top1_validate = {}, best_top5_validate = {}, "
                  "top1_validate = {}, top5_validate = {}, class_loss_validate = {}"
                  .format(filepath, checkpoint['epoch'], self.best_top1_validate, self.best_top5_validate,
                          checkpoint['top1_validate'], checkpoint['top5_validate'], checkpoint['class_loss_validate']))
        else:
            print("=> no checkpoint found at '{}'".format(filepath))

