import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from ..SOF_VSR import SOF_VSR
from ..models import SOFVSR, OFRnet, SRnet


# gpu mode
# config
# why does this paper not need  any epochs?

class TrainModel(SOF_VSR):
    def __init__(self, scale):
        super(TrainModel, self).__init__(scale=scale)
        self.SOF_VSR_net = SOFVSR(cfg, is_training=True)

        # SHARDUL PLACEHOLDER
        # dataloader
        self.train_set = TrainsetLoader(cfg)
        self.train_loader = DataLoader(self.train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)
        ##########

        self.optimizer = torch.optim.Adam(self.SOF_VSR_net.parameters(), lr=1e-3)
        self.milestones = [80000, 160000]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1)

        self.criterion = torch.nn.MSELoss()
        self.loss_list = []

    def get_models(self):
        pass

    def __call__(self, *args, **kwargs):
        for idx_iter, (LR, HR) in enumerate(self.train_loader):
            self.scheduler.step()

            # data
            b, n_frames, h_lr, w_lr = LR.size()
            idx_center = (n_frames - 1) // 2

            LR, HR = Variable(LR), Variable(HR)
            if cfg.gpu_mode:
                LR = LR.cuda()
                HR = HR.cuda()
            LR = LR.view(b, -1, 1, h_lr, w_lr)
            HR = HR.view(b, -1, 1, h_lr * cfg.scale, w_lr * cfg.scale)

            # inference
            flow_L1, flow_L2, flow_L3, SR = self.SOF_VSR_net(LR)

            # loss
            loss_SR = self.criterion(SR, HR[:, idx_center, :, :, :])
            loss_OFR = torch.zeros(1).cuda()

            for i in range(n_frames):
                if i != idx_center:
                    loss_L1 = self.OFR_loss(F.avg_pool2d(LR[:, i, :, :, :], kernel_size=2),
                                            F.avg_pool2d(LR[:, idx_center, :, :, :], kernel_size=2),
                                            flow_L1[i])
                    loss_L2 = self.OFR_loss(LR[:, i, :, :, :], LR[:, idx_center, :, :, :], flow_L2[i])
                    loss_L3 = self.OFR_loss(HR[:, i, :, :, :], HR[:, idx_center, :, :, :], flow_L3[i])
                    loss_OFR = loss_OFR + loss_L3 + 0.2 * loss_L2 + 0.1 * loss_L1

            loss = loss_SR + 0.01 * loss_OFR / (n_frames - 1)
            loss_list.append(loss.data.cpu())

            # backwards
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # save checkpoint
            # SHARDUL TODO CHECKPOINTING CODE AND DISPLAY
            if idx_iter % 5000 == 0:
                print('Iteration---%6d,   loss---%f' % (idx_iter + 1, np.array(loss_list).mean()))
                save_path = 'log/' + cfg.degradation + '_x' + str(cfg.scale)
                save_name = cfg.degradation + '_x' + str(cfg.scale) + '_iter' + str(idx_iter) + '.pth'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(net.state_dict(), save_path + '/' + save_name)
                loss_list = []

            # SHARDUL TODO DECIDE WHAT TO RETURN
