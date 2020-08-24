import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import os

from .adacof import AdaCoF
from ..models import AdaCoFNet
from torchOnVideo.losses import AdaCoF_loss
from torchOnVideo.datasets.Vimeo90KTriplet.frame_interpolation import TrainAdaCoF
from torchOnVideo.frame_interpolation.utils import to_variable

parser = argparse.ArgumentParser(description='AdaCoF-Pytorch')

# parameters
# Model Selection
parser.add_argument('--model', type=str, default='adacofnet')

# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--train', type=str, default='./db/vimeo_triplet')
parser.add_argument('--out_dir', type=str, default='./output_adacof_train')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--test_input', type=str, default='./test_input/middlebury_others/input')
parser.add_argument('--gt', type=str, default='./test_input/middlebury_others/gt')

# Learning Options
parser.add_argument('--epochs', type=int, default=50, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--loss', type=str, default='1*Charb+0.01*g_Spatial+0.005*g_Occlusion', help='loss function configuration')
parser.add_argument('--patch_size', type=int, default=256, help='Patch size')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAMax', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'), help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Options for AdaCoF
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

# transform = transforms.Compose([transforms.ToTensor()])

# left out GPU ID
# If train Loader is set to None
# Use the one present in the default location
# We will provide both batch size if train_loader is not provided
class TrainModel(AdaCoF):
    def __init__(self, model=None, train_set=None, train_dir='../../db/Vimeo90K_Triplet_AdaCoF', random_crop = None,resize = None, augment_s=True, augment_t=True,
                 train_data_loader = None, loss=None, checkpoint=None, start_epoch = 0, use_start_epoch_checkpoint=False,
                 output_dir = "../../outputs/Vimeo90K_Triplet_AdaCoF",
                 epochs = 50, batch_size=4, patch_size=256, shuffle = True, num_workers=0,
                 optimizer = None, lr=0.001,
                 scheduler = None,
                 kernel_size=5, dilation=1,
                 epoch_display_step=1, batch_display_step=1):
        super(TrainModel, self).__init__()
        # self.args = args

        # self.train_loader = train_loader
        # self.max_step = self.train_loader.__len__()
        # self.test_loader = test_loader
        # self.model = my_model

        # Shardul Main part -- loss is left
        if loss in None:
            self.loss = AdaCoF_loss()
        else:
            self.loss = loss

        # self.current_epoch = start_epoch

        # give the expected location too
        if train_set is None:
            self.train_set = TrainAdaCoF(train_dir, random_crop=random_crop, resize=resize, augment_s=augment_s, augment_t=augment_t)
        else:
            self.train_set = train_set

        if train_data_loader is None:
            self.train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            self.train_loader = train_data_loader

        self.max_step = self.train_loader.__len__()

        if model is None:
            self.model = AdaCoFNet(kernel_size, dilation)
        else:
            self.model = model

        if checkpoint is not None:
            checkpoint_f = torch.load(checkpoint)
            self.model.load(checkpoint_f['state_dict'])
            start_epoch_checkpoint = checkpoint['epoch']

        if use_start_epoch_checkpoint is False:
            self.start_epoch = start_epoch
        else:
            self.start_epoch = start_epoch

        if optimizer is None:
            self.optimizer = optim.Adamax(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        else:
            self.scheduler = scheduler

        # self.optimizer = utility.make_optimizer(args, self.model)
        # self.scheduler = utility.make_scheduler(args, self.optimizer)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.result_dir = os.path.join(output_dir, 'result')
        self.ckpt_dir = os.path.join(output_dir, 'checkpoint')

        self.total_epochs = epochs

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.logfile = open(os.path.join(output_dir, 'log.txt'), 'w')

        self.epoch_display_step = epoch_display_step
        self.batch_display_step = batch_display_step

    def __call__(self, *args, **kwargs):
        self.model.train()
        for running_epoch in range(self.start_epoch, self.total_epochs):
            for batch_idx, (frame0, frame1, frame2) in enumerate(self.train_loader):
                frame0 = to_variable(frame0)
                frame1 = to_variable(frame1)
                frame2 = to_variable(frame2)

                self.optimizer.zero_grad()

                output = self.model(frame0, frame2)

                # currently uses only AdaCoF loss
                loss = self.loss(output, frame1, [frame0, frame2])
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.batch_display_step == 0:
                    print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ',
                                                                                '[' + str(running_epoch + 1) + '/' + str(
                                                                                    self.total_epochs) + ']', 'Step: ',
                                                                                '[' + str(batch_idx) + '/' + str(
                                                                                    self.max_step) + ']', 'train loss: ',
                                                                                loss.item()))
            self.scheduler.step()