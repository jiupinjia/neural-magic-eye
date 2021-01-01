import numpy as np
import matplotlib.pyplot as plt
import os

from networks import *
import utils

import torch
torch.cuda.current_device()
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Decoder():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders

        self.net_G = define_G(args).to(device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        self.optimizer_G = optim.Adam(
            self.net_G.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # define lr schedulers
        self.exp_lr_scheduler_G = lr_scheduler.StepLR(
            self.optimizer_G, step_size=args.scheduler_step_size, gamma=0.1)

        # define some other vars to record the training states
        self.running_acc = []
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_num_epochs
        self.G_pred = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # define loss functions
        self._loss = nn.MSELoss()

        # buffers to save training/val accuracy
        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self):

        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            print('loading last checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'))

            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])
            self.net_G.to(device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            print('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            print()

        else:
            print('training from scratch...')


    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict()
        }, os.path.join(self.checkpoint_dir, ckpt_name))


    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()


    def _compute_acc(self):

        target = self.batch['dmap'].to(device).detach()
        img = self.G_pred.detach()
        psnr = utils.cpt_batch_psnr(img, target, PIXEL_MAX=1.0)

        return psnr


    def _collect_running_batch_states(self):
        self.running_acc.append(self._compute_acc().item())

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        if np.mod(self.batch_id, 10) == 1:
            print('Is_training: %s. [%d,%d][%d,%d], G_loss: %.5f, running_acc: %.5f'
                  % (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     self.G_loss.item(), np.mean(self.running_acc)))


    def _visualize_batch_and_prediction(self):

        if np.mod(self.batch_id, 100) == 1:
            vis_input = utils.make_numpy_grid(self.batch['stereogram'])
            vis_pred = utils.make_numpy_grid(self.G_pred)
            vis_gt = utils.make_numpy_grid(self.batch['dmap'])
            vis = np.concatenate([vis_input, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)


    def _collect_epoch_states(self):

        self.epoch_acc = np.mean(self.running_acc)
        print('Is_training: %s. Epoch %d / %d, epoch_acc= %.5f' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        print()


    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        print('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        print()

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            print('*' * 10 + 'Best model updated!')
            print()


    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)


    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)


    def _clear_cache(self):
        self.running_acc = []


    def _forward_pass(self, batch):
        self.batch = batch
        img_in = batch['stereogram'].to(device)
        self.G_pred =self.net_G(img_in)


    def _backward_G(self):

        gt = self.batch['dmap'].to(device)
        self.G_loss = self._loss(self.G_pred, gt)
        self.G_loss.backward()


    def train_models(self):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._visualize_batch_and_prediction()
            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()

            ################## Eval ##################
            ##########################################
            print('Begin evaluation...')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
                self._visualize_batch_and_prediction()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()


class Classifier(Decoder):

    def __init__(self, args, dataloaders):
        super(Classifier, self).__init__(args, dataloaders)

        if args.dataset not in ['mnist', 'shapenet']:
            raise NotImplementedError(
                'Wrong dataset name %s (classification mode only supports [mnist] and [shapenet] dataset)'
                % args.dataset)

        self._loss = nn.CrossEntropyLoss()
        self.prediction_mode = args.prediction_mode

    def _compute_acc(self):
        target = self.batch['label'].to(device).detach()
        predicted = self.G_pred.detach()
        cls_acc = utils.cpt_batch_classification_acc(predicted, target)
        return cls_acc

    def _forward_pass(self, batch):

        self.batch = batch
        if self.prediction_mode == 'stereogram2label':
            # predict label from autostereogram
            img_in = batch['stereogram'].to(device)
        elif self.prediction_mode == 'depth2label':
            # predict label from depthmap (upperbound model)
            img_in = batch['dmap'].to(device)
            img_in = img_in.repeat((1, 3, 1, 1))
        else:
            raise NotImplementedError(
                'Wrong prediction mode %s (choose one from [stereogram2label] or [depth2label])'
                % self.prediction_mode)
        self.G_pred =self.net_G(img_in)


    def _backward_G(self):
        target = self.batch['label'].to(device).long()
        self.G_loss = self._loss(self.G_pred, target)
        self.G_loss.backward()

    def _visualize_batch_and_prediction(self):
        pass


