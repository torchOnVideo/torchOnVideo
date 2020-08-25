import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import os


from .cain import CAIN
from torchOnVideo.datasets.Vimeo90KTriplet.frame_interpolation import TrainCAIN
from torchOnVideo.frame_interpolation.models import CAIN, CAIN_NoCA, CAIN_EncDec
from torchOnVideo.frame_interpolation.utils import build_input_CAIN


class TrainModel(CAIN):
    def __init__(self, model=None, cain_model=None, train_set=None, train_dir='../../db/Vimeo90K_Triplet_CAIN', random_crop = None,resize = None, augment_s=True, augment_t=True,
                 train_data_loader = None, loss=None, checkpoint=None, start_epoch = 0, use_start_epoch_checkpoint=False,
                 output_dir = "../../outputs/Vimeo90K_Triplet_CAIN",
                 epochs=200, batch_size=16, shuffle = True, num_workers=0,
                 optimizer = None, lr=0.0002,
                 scheduler = None,
                 depth=3,
                 epoch_display_step=1, batch_display_step=1,
                 use_dataparallel=False, pin_memory=True):
        super(TrainModel, self).__init__()

        if loss in None:
            self.loss = nn.L1Loss()

        # self.current_epoch = start_epoch

        self.depth = depth

        print('==> Building model ')
        if cain_model is not None:
            if cain_model == 'cain':
                self.model = CAIN(depth=self.depth)
            elif cain_model == 'cain_encdec':
                self.model = CAIN_EncDec(depth=self.depth, start_filts=32)
            elif cain_model == 'cain_noca':
                self.model = CAIN_NoCA(depth = self.depth)
            else:
                raise NotImplementedError("Unknown cain model")
        elif model is not None:
            self.model = model
        else:
            self.model = CAIN(depth=self.depth)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if use_dataparallel is True:
            self.model = torch.nn.DataParallel(self.model).to(device)

        print('==> Building training set ')
        # give the expected location too
        if train_set is None:
            self.train_set = TrainCAIN(data_root=train_dir)
        else:
            self.train_set = train_set


        print('==> Building training data loader ')
        if train_data_loader is None:
            self.train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
        else:
            self.train_loader = train_data_loader

        self.max_step = self.train_loader.__len__()

        if checkpoint is not None:
            checkpoint_f = torch.load(checkpoint)
            self.model.load(checkpoint_f['state_dict'])
            start_epoch_checkpoint = checkpoint['epoch']

        if use_start_epoch_checkpoint is False:
            self.start_epoch = start_epoch
        else:
            self.start_epoch = start_epoch

        print('==> Building optimizer ')
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        print('==> Building scheduler ')
        if scheduler is None:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min', factor=0.5, patience=5, verbose=True)
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
        print('==> Training has started ')
        for running_epoch in range(self.start_epoch, self.total_epochs):
            for i, (images, imgpaths) in enumerate(self.train_loader):

                # Build input batch
                im1, im2, gt = build_input_CAIN(images, imgpaths)

                # Forward
                self.optimizer.zero_grad()
                out, feats = self.model(im1, im2)
                loss = self.loss(out, gt)

                # Save loss values
                # currently will not use these losses
                # for k, v in losses.items():
                #     if k != 'total':
                #         v.update(loss_specific[k].item())
                # if LOSS_0 == 0:
                #     LOSS_0 = loss.data.item()
                # losses['total'].update(loss.item())

                loss.backward()
                # shardul comment
                # if loss.data.item() > 10.0 * LOSS_0:
                #     print(max(p.grad.data.abs().max() for p in model.parameters()))
                #     continue
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                self.optimizer.step()

                # Calc metrics & print log
                # Shardul check printing logging metrics
                # if i % args.log_iter == 0:
                #     utils.eval_metrics(out, gt, psnrs, ssims, lpips, lpips_model)
                #
                #     print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tPSNR: {:.4f}\tTime({:.2f})'.format(
                #         epoch, i, len(train_loader), losses['total'].avg, psnrs.avg, time.time() - t))
                #
                #     # Log to TensorBoard
                #     utils.log_tensorboard(writer, losses, psnrs.avg, ssims.avg, lpips.avg,
                #                           optimizer.param_groups[-1]['lr'], epoch * len(train_loader) + i)
                #
                #     # Reset metrics
                #     losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
                #     t = time.time()

            if i % self.batch_display_step == 0:
                    print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ',
                                                                                '[' + str(running_epoch + 1) + '/' + str(
                                                                                    self.total_epochs) + ']', 'Step: ',
                                                                                '[' + str(i) + '/' + str(
                                                                                    self.max_step) + ']', 'train loss: ',
                                                                                loss.item()))

            # update optimizer policy
            self.scheduler.step(loss)
            ############


