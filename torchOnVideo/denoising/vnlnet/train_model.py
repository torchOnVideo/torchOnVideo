import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os

from .vnlnet import VNLNet
from ..models import ModifiedDnCNN
from tensorboardX import SummaryWriter
from .utils import weights_init_kaiming, batch_PSNR

from torchOnVideo.datasets.f16_video_dataset.denoising import TrainVNLNet


class TrainModel(VNLNet):
    def __init__(self, model=None, train_set=None, train_dir='../../db/f16_vnlnet_traindata', random_crop=None,
                 resize=None,
                 augment_s=True, augment_t=True,
                 train_data_loader=None, loss=None, checkpoint=None, start_epoch=0, use_start_epoch_checkpoint=False,
                 output_dir="../../outputs/f16vnlnet",
                 epochs=20, batch_size=128, shuffle=True, num_workers=2,
                 optimizer=None, lr=0.001, milestone=[12, 17],
                 scheduler=None,
                 sigma=20, color=True, nn_patch_width=41, pass_nn_value=True, oracle_mode=0, search_window_width=41,
                 past_frames=7, future_frames=7,
                 epoch_display_step=1, batch_display_step=1,
                 run_validation=False, val_dir="../../db/f16_vnlnet_valdata", val_set=None, val_loader=None,
                 save_dir="../../db/f_16_vnlnet_save_dir"):

        self.sigma = sigma / 255.
        self.color = color
        self.nn_patch_width = nn_patch_width
        self.pass_nn_value = pass_nn_value
        self.oracle_mode = oracle_mode
        self.search_window_width = search_window_width
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.milestone = milestone
        self.batch_display_step = batch_display_step
        self.run_validation = run_validation
        self.batch_size = batch_size
        self.save_dir = save_dir

        print('==> Building training set ')
        if train_set is None:
            self.train_set = TrainVNLNet(train_dir, color_mode=self.color, sigma=self.sigma,
                                         oracle_mode=self.oracle_mode, past_frames=self.past_frames,
                                         future_frames=self.future_frames,
                                         search_window_width=self.search_window_width,
                                         nn_patch_width=self.nn_patch_width,
                                         pass_nn_value=self.pass_nn_value)
        else:
            self.train_set = train_set

        print('==> Building training data loader ')
        if train_data_loader is None:
            self.train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers)
        else:
            self.train_loader = train_data_loader

        self.input_channels = self.train_set.data_num_channels()
        self.output_channels = (3 if self.color else 1)
        self.nlconv_features = (96 if self.color else 32)
        self.nlconv_layers = 4
        self.dnnconv_features = (192 if self.color else 64)
        self.dnnconv_layers = 15

        print('==> Building model ')
        if model is None:
            self.model = ModifiedDnCNN(input_channels=self.input_channels,
                                       output_channels=self.output_channels,
                                       nlconv_features=self.nlconv_features,
                                       nlconv_layers=self.nlconv_layers,
                                       dnnconv_features=self.dnnconv_features,
                                       dnnconv_layers=self.dnnconv_layers)
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

        print("==>Building Loss")
        if loss in None:
            self.criterion = nn.MSELoss(size_average=False)
        else:
            self.criterion = loss

        self.max_step = self.train_loader.__len__()

        print("==>Building Validation Components")
        if run_validation:
            if val_set is None:
                self.val_set = TrainVNLNet(val_dir, color_mode=self.color, sigma=self.sigma,
                                           oracle_mode=self.oracle_mode, past_frames=self.past_frames,
                                           future_frames=self.future_frames,
                                           search_window_width=self.search_window_width,
                                           nn_patch_width=self.nn_patch_width,
                                           pass_nn_value=self.pass_nn_value)
            else:
                self.val_set = val_set

            if val_loader is None:
                self.val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=shuffle,
                                             num_workers=num_workers)
            else:
                self.val_loader = val_loader

        device = torch.device("cuda:0")
        model.to(device)
        self.criterion.cuda()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.writer = SummaryWriter(self.save_dir)

    def __call__(self, *args, **kwargs):
        self.model.train()

        for running_epoch in range(self.start_epoch, self.total_epochs):

            # train over all data in the epoch
            for i, data in enumerate(self.train_loader, 0):
                # Pre-training step
                self.model.train()
                self.model.zero_grad()
                self.optimizer.zero_grad()

                (stack_train, expected_train) = data

                stack_train = Variable(stack_train.cuda(), volatile=True)
                expected_train = Variable(expected_train.cuda(), volatile=True)

                # Evaluate model and optimize it
                out_train = self.model(stack_train)
                loss = self.criterion(out_train, expected_train) / (expected_train.size()[0] * 2)
                loss.backward()
                self.optimizer.step()

                # Shardul Check Evaluation component, saving and logging

                if i % 10 == 0:
                    # Results
                    self.model.eval()
                    psnr_train = batch_PSNR(expected_train, out_train, 1.)
                    print('[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f' % \
                          (running_epoch + 1, i + 1, len(self.train_loader), loss.item(), psnr_train))
                    # Log the scalar values
                    self.writer.add_scalar('loss', loss.item(), i)
                    self.writer.add_scalar('PSNR on training data', psnr_train, i)

            if self.run_validation:
                self.model.eval()
                psnr_val = 0
                with torch.no_grad():
                    for i, data in enumerate(self.val_loader, 0):
                        (stack_val, expected_val) = data
                        stack_val = Variable(stack_val.cuda())
                        expected_val = Variable(expected_val.cuda())
                        out_val = self.model(stack_val)
                        psnr_val += batch_PSNR(out_val, expected_val, 1.)
                psnr_val /= len(self.val_set)
                psnr_val *= self.batch_size

                print('\n[epoch %d] PSNR_val: %.4f' % (running_epoch + 1, psnr_val))
                self.writer.add_scalar('PSNR on validation data', psnr_val, running_epoch)
                # writer.add_scalar('Learning rate', current_lr, epoch)

            net_data = { \
                'model_state_dict': self.model.state_dict(), \
                'args': args \
                }
            torch.save(net_data, os.path.join(args.save_dir, 'net.pth'))

            # Prepare next epoch
            self.train_set.prepare_epoch()
            self.scheduler.step()
