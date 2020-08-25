import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os

from ..SOF_VSR import SOF_VSR
from ..models import SOFVSR, OFRnet, SRnet
from torchOnVideo.datasets.CVDL.super_resolution import TrainSOFVSR
from torchOnVideo.losses import OFR_loss


class TrainModel(SOF_VSR):
    def __init__(self, model=None, train_set=None, train_dir='../../db/CVDL_SOFVSR_traindata',
                 train_data_loader=None, loss=None, checkpoint=None, start_epoch=0, use_start_epoch_checkpoint=False,
                 output_dir="../../outputs/CVDL_SOFVSR",
                 scale = 4, patch_size=32, degradation='BI',
                 epochs=20, batch_size=32, shuffle=True, num_workers=4,
                 n_iters=200000,
                 optimizer=None, lr=1e-3, milestone=[80000, 16000],
                 scheduler=None, gpu_mode=False,
                 epoch_display_step=1, batch_display_step=1,
                 run_validation=False, val_dir="../../db/f16_vnlnet_valdata", val_set=None, val_loader=None):

        super(TrainModel, self).__init__(scale=scale)

        self.degradation = degradation
        self.gpu_mode = gpu_mode

        print('==> Building training set ')
        if train_set is None:
            self.train_set = TrainSOFVSR(trainset_dir=train_dir, scale=scale, patch_size=patch_size, n_iters=n_iters,
                                         batch_size=batch_size, degradation=degradation)
        else:
            self.train_set = train_set


        print('==> Building training data loader ')
        if train_data_loader is None:
            self.train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers)
        else:
            self.train_loader = train_data_loader

        print('==> Building model ')
        if model is None:
            self.model = SOFVSR(scale=scale)
        else:
            self.model = model

        print('==> Building optimizer ')
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        print('==> Building scheduler ')
        if scheduler is None:
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestone, gamma=0.01)
        else:
            self.scheduler = scheduler

        if loss in None:
            self.criterion = nn.MSELoss(size_average=False)
        else:
            self.criterion = loss

        self.max_step = self.train_loader.__len__()

    def __call__(self, *args, **kwargs):
        self.model.train()
        print('==> Training has started ')
        loss_list = []
        for idx_iter, (LR, HR) in enumerate(self.train_loader):
            self.scheduler.step()

            # data
            b, n_frames, h_lr, w_lr = LR.size()
            idx_center = (n_frames - 1) // 2

            LR, HR = Variable(LR), Variable(HR)


            if self.gpu_mode:
                LR = LR.cuda()
                HR = HR.cuda()
            LR = LR.view(b, -1, 1, h_lr, w_lr)
            HR = HR.view(b, -1, 1, h_lr * self.scale, w_lr * self.scale)

            # inference
            flow_L1, flow_L2, flow_L3, SR = self.model(LR)

            # loss
            loss_SR = self.criterion(SR, HR[:, idx_center, :, :, :])

            # SHARDUL CHECK CUDA
            loss_OFR = torch.zeros(1).cuda()

            for i in range(n_frames):
                if i != idx_center:
                    loss_L1 = OFR_loss(F.avg_pool2d(LR[:, i, :, :, :], kernel_size=2),
                                       F.avg_pool2d(LR[:, idx_center, :, :, :], kernel_size=2),
                                       flow_L1[i])
                    loss_L2 = OFR_loss(LR[:, i, :, :, :], LR[:, idx_center, :, :, :], flow_L2[i])
                    loss_L3 = OFR_loss(HR[:, i, :, :, :], HR[:, idx_center, :, :, :], flow_L3[i])
                    loss_OFR = loss_OFR + loss_L3 + 0.2 * loss_L2 + 0.1 * loss_L1

            loss = loss_SR + 0.01 * loss_OFR / (n_frames - 1)
            loss_list.append(loss.data.cpu())

            # backwards
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save checkpoint
            if idx_iter % 5000 == 0:
                print('Iteration---%6d,   loss---%f' % (idx_iter + 1, np.array(loss_list).mean()))
                save_path = 'log/' + self.degradation + '_x' + str(self.scale)
                save_name = self.degradation + '_x' + str(self.scale) + '_iter' + str(idx_iter) + '.pth'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(self.model.state_dict(), save_path + '/' + save_name)
                loss_list = []


