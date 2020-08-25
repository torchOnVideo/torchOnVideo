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

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=False)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=8, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./vimeo_septuplet/sequences')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
# parser.add_argument('--pretrained_sr', default='3x_dl10VDBPNF7_epoch_84.pth', help='sr pretrained base model')
parser.add_argument('--pretrained_sr', default='RBPN_4x_F11_NTIRE2019.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')


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
                 scheduler=None,
                 epoch_display_step=1, batch_display_step=1, num_gpus = 1,
                 run_validation=False, val_dir="../../db/f16_vnlnet_valdata", val_set=None, val_loader=None):
        super(TrainModel, self).__init__(scale=scale)

        # Shardul GPU Check
        # opt = parser.parse_args()
        # gpus_list = range(opt.gpus)
        # hostname = str(socket.gethostname())
        # cudnn.benchmark = True
        # print(opt)
        #
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            self.model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5,
                              nFrames=self.nFrames, scale_factor=self.upscale_factor)
        else:
            self.model = model

        # Shardul check GPU
        # self.gpu_list = range(num_gpus)
        # self.model = torch.nn.DataParallel(model, device_ids=self.gpu_list)

        # if cuda:
        #     model = model.cuda(gpus_list[0])
        #     criterion = criterion.cuda(gpus_list[0])

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

        print("Building Loss")
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
                #t0 = time.time()
                prediction = self.model(input, neigbor, flow)

                if self.residual:
                    prediction = prediction + bicubic

                loss = self.criterion(prediction, target)
                # import pdb; pdb.set_trace()
                epoch_loss += loss.item()
                # epoch_loss += loss.data[0]
                loss.backward()
                self.optimizer.step()


                # Shardul check saving and display
            #     print("==> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
            #                                                                             len(training_data_loader),
            #                                                                             loss.item(), (t1 - t0)))
            #
            # print("==> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
            #
            #

