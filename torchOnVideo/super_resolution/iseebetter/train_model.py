import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os

from .iseebetter import ISeeBetter
from torchOnVideo.datasets.Vimeo90KSeptuplet.super_resolution import TrainISeeBetter
from torchOnVideo.super_resolution.models import RBPN


class TrainModel(ISeeBetter):
    def __init__(self, model=None, train_set=None, train_dir='../../db/Vimeo90K_septuplet_traindata',
                 file_list="sep_trainlist.txt",
                 train_data_loader=None, loss=None, checkpoint=None, start_epoch=0, use_start_epoch_checkpoint=False,
                 other_dataset=False, future_frame=True,
                 output_dir="../../outputs/CVDL_SOFVSR",
                 scale=4, nFrames=7, patch_size=64,
                 epochs=150, batch_size=32, shuffle=True, num_workers=4, residual=False,
                 data_augmentation=True,
                 optimizer=None, lr=1e-4, milestone=[75, 150],
                 scheduler=None, use_gpu=False,
                 epoch_display_step=1, batch_display_step=1, num_gpus=1,
                 run_validation=False, val_dir="../../db/f16_vnlnet_valdata", val_set=None, val_loader=None):
        super(TrainModel, self).__init__(scale=scale)

        print('==> Building training set ')
        if train_set is None:
            self.train_set = TrainISeeBetter(image_dir=os.path.join(train_dir, "sequences"), nFrames=nFrames,
                                             upscale_factor=self.scale, data_augmentation=data_augmentation,
                                             file_list=file_list, other_dataset=other_dataset, patch_size=patch_size,
                                             future_frame=future_frame)
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
            self.model = RBPN(num_channels=3, base_filter=256, feat=64, num_stages=3, n_resblock=5,
                              nFrames=self.nFrames, scale_factor=self.upscale_factor)
        else:
            self.model = model

        self.gpu_list = range(num_gpus)
        self.model = torch.nn.DataParallel(model, device_ids=self.gpu_list)

        if use_gpu:
            self.model = model.cuda(self.gpus_list[0])
            self.criterion = self.criterion.cuda(self.gpus_list[0])

        print('==> Building optimizer ')
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        print('==> Building scheduler ')
        if scheduler is None:
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestone, gamma=0.1)
        else:
            self.scheduler = scheduler

        print("==>Building Loss")
        if loss in None:
            self.criterion = nn.L1Loss()
        else:
            self.criterion = loss

        self.max_step = self.train_loader.__len__()
        self.total_epochs = epochs
        self.residual = residual

    def __call__(self, *args, **kwargs):
        self.model.train()
        epoch_loss = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('==> Training has started ')
        for running_epoch in range(self.start_epoch, self.total_epochs):
            for iteration, batch in enumerate(self.train_loader, 1):
                # import pdb; pdb.set_trace()

                input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]

                # Shardul check device
                input = Variable(input).to(device=device, dtype=torch.float)
                bicubic = Variable(bicubic).to(device=device, dtype=torch.float)
                neigbor = [Variable(j).to(device=device, dtype=torch.float) for j in neigbor]
                flow = [Variable(j).to(device=device, dtype=torch.float) for j in flow]

                self.optimizer.zero_grad()
                # t0 = time.time()
                prediction = self.model(input, neigbor, flow)

                if self.residual:
                    prediction = prediction + bicubic

                loss = self.criterion(prediction, target)
                # import pdb; pdb.set_trace()
                epoch_loss += loss.item()
                # epoch_loss += loss.data[0]
                loss.backward()
                self.optimizer.step()

                print("==> Epoch[{}]({}/{}): Loss: {:.4f} ".format(running_epoch, iteration,
                                                                   len(self.train_loader),
                                                                   loss.item()))

                print("==> Epoch {} Complete: Avg. Loss: {:.4f}".format(running_epoch,
                                                                        epoch_loss / len(self.train_loader)))
