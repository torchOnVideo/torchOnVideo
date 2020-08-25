import argparse
from tqdm import tqdm
import os
import pandas as pd

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from torchOnVideo.data import get_training_set, get_eval_set
from torchOnVideo.losses import ISeeBetterLoss

from torch.utils.data import DataLoader

from .models import RBPN, SRGAN_Discriminator

# FIXME Deprecated Remove
from torch.autograd import Variable

import gc

UPSCALE_FACTOR = 4

parser = argparse.ArgumentParser(description='Train iSeeBetter: Super Resolution Models')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
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
parser.add_argument('--pretrained_sr', default='RBPN_4x.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')
parser.add_argument('--APITLoss', action='store_true', help='Use APIT Loss')
parser.add_argument('--useDataParallel', action='store_true', help='Use DataParallel')
parser.add_argument('-v', '--debug', default=False, action='store_true', help='Print debug spew.')

# When the class is called - we can either just use the models with get_models function
# use train method either with default values or use a json file to change the params
# use the inference method directly if you have a pretrained model
class iSeeBetter():
    """
        Paper Name: iSeeBetter: Spatio-Temporal Video Super-Resolution using Recurrent Generative Back-Projection Networks
        Authors: A. Chadha, J. Britto and M. M. Roja
    """

    def __init__(self):
        args = parser.parse_args()
        self.args = args

        train_set = get_training_set(args.data_dir, args.nFrames, args.upscale_factor, args.data_augmentation,
                                     args.file_list,
                                     args.other_dataset, args.patch_size, args.future_frame)
        self.training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize,
                                          shuffle=True)

        self.netG = RBPN(num_channels=3, base_filter=256, feat=64, num_stages=3, n_resblock=5, nFrames=args.nFrames,
                    scale_factor=args.upscale_factor)

        if args.useDataParallel:
            gpus_list = range(args.gpus)
            self.netG = torch.nn.DataParallel(self.netG, device_ids=gpus_list)

        self.netD = SRGAN_Discriminator()

        self.generatorCriterion = nn.L1Loss() if not args.APITLoss else ISeeBetterLoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu_mode else "cpu")

        self.netG.to(self.device)
        self.netD.to(self.device)

        self.generatorCriterion.to(self.device)

        #FIXME add option from argumnets or json for optimizer
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    def get_models(self):
        """

        :return: all the models used in iSeeBetter - RBPN and the SRGAN discriminator
        """
        return self.netG, self.netD

    def saveModelParams(self, epoch, runningResults):
                        #, netG, netD):
        results = {'DLoss': [], 'GLoss': [], 'DScore': [], 'GScore': [], 'PSNR': [], 'SSIM': []}

        # Save model parameters
        torch.save(self.netG.state_dict(), 'weights/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(self.netD.state_dict(), 'weights/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

        # logger.info("Checkpoint saved to {}".format('weights/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))
        # logger.info("Checkpoint saved to {}".format('weights/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))

        # Save Loss\Scores\PSNR\SSIM
        results['DLoss'].append(runningResults['DLoss'] / runningResults['batchSize'])
        results['GLoss'].append(runningResults['GLoss'] / runningResults['batchSize'])
        results['DScore'].append(runningResults['DScore'] / runningResults['batchSize'])
        results['GScore'].append(runningResults['GScore'] / runningResults['batchSize'])
        # results['PSNR'].append(validationResults['PSNR'])
        # results['SSIM'].append(validationResults['SSIM'])

        if epoch % 1 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'DLoss': results['DLoss'], 'GLoss': results['GLoss'], 'DScore': results['DScore'],
                      'GScore': results['GScore']},  # , 'PSNR': results['PSNR'], 'SSIM': results['SSIM']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'iSeeBetter_' + str(UPSCALE_FACTOR) + '_Train_Results.csv',
                              index_label='Epoch')

    def trainModel(self):
        for epoch in range(self.args.start_epoch, self.args.nEpochs + 1):
            # runningResults =self.train_single_epoch(epoch, self.training_data_loader, self.netG, self.netD, optimizerD, optimizerG,
            #                             self.generatorCriterion, self.device, self.args)
            runningResults = self.train_single_epoch(epoch)

            if (epoch + 1) % (self.args.snapshots) == 0:
                # self.saveModelParams(epoch, runningResults, netG, netD)
                self.saveModelParams(epoch, runningResults)

    def train_single_epoch(self, epoch):
        # def trainModel(epoch, training_data_loader, netG, netD, optimizerD, optimizerG, generatorCriterion, device,
        #                args):
            trainBar = tqdm(self.training_data_loader)
            runningResults = {'batchSize': 0, 'DLoss': 0, 'GLoss': 0, 'DScore': 0, 'GScore': 0}

            self.netG.train()
            self.netD.train()

            # Skip first iteration
            iterTrainBar = iter(trainBar)
            next(iterTrainBar)

            for data in iterTrainBar:
                batchSize = len(data)
                runningResults['batchSize'] += batchSize

                ################################################################################################################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ################################################################################################################
                if self.args.APITLoss:
                    fakeHRs = []
                    targets = []
                fakeScrs = []
                realScrs = []

                DLoss = 0

                # Zero-out gradients, i.e., start afresh
                self.netD.zero_grad()

                input, target, neigbor, flow, bicubic = data[0], data[1], data[2], data[3], data[4]
                if self.args.gpu_mode and torch.cuda.is_available():
                    input = Variable(input).cuda()
                    target = Variable(target).cuda()
                    bicubic = Variable(bicubic).cuda()
                    neigbor = [Variable(j).cuda() for j in neigbor]
                    flow = [Variable(j).cuda().float() for j in flow]
                else:
                    input = Variable(input).to(device=self.device, dtype=torch.float)
                    target = Variable(target).to(device=self.device, dtype=torch.float)
                    bicubic = Variable(bicubic).to(device=self.device, dtype=torch.float)
                    neigbor = [Variable(j).to(device=self.device, dtype=torch.float) for j in neigbor]
                    flow = [Variable(j).to(device=self.device, dtype=torch.float) for j in flow]

                fakeHR = self.netG(input, neigbor, flow)
                if self.args.residual:
                    fakeHR = fakeHR + bicubic

                realOut = self.netD(target).mean()
                fakeOut = self.netD(fakeHR).mean()

                if self.args.APITLoss:
                    fakeHRs.append(fakeHR)
                    targets.append(target)
                fakeScrs.append(fakeOut)
                realScrs.append(realOut)

                DLoss += 1 - realOut + fakeOut

                DLoss /= len(data)

                # Calculate gradients
                DLoss.backward(retain_graph=True)

                # Update weights
                self.optimizerD.step()

                ################################################################################################################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ################################################################################################################
                GLoss = 0

                # Zero-out gradients, i.e., start afresh
                self.netG.zero_grad()

                if self.args.APITLoss:
                    idx = 0
                    for fakeHR, fake_scr, HRImg in zip(fakeHRs, fakeScrs, targets):
                        fakeHR = fakeHR.to(self.device)
                        fake_scr = fake_scr.to(self.device)
                        HRImg = HRImg.to(self.device)
                        GLoss += self.generatorCriterion(fake_scr, fakeHR, HRImg, idx)
                        idx += 1
                else:
                    GLoss = self.generatorCriterion(fakeHR, target)

                GLoss /= len(data)

                # Calculate gradients
                GLoss.backward()

                # Update weights
                self.optimizerG.step()

                realOut = torch.Tensor(realScrs).mean()
                fakeOut = torch.Tensor(fakeScrs).mean()
                runningResults['GLoss'] += GLoss.item() * self.args.batchSize
                runningResults['DLoss'] += DLoss.item() * self.args.batchSize
                runningResults['DScore'] += realOut.item() * self.args.batchSize
                runningResults['GScore'] += fakeOut.item() * self.args.batchSize

                trainBar.set_description(desc='[Epoch: %d/%d] D Loss: %.4f G Loss: %.4f D(x): %.4f D(G(z)): %.4f' %
                                              (epoch, self.args.nEpochs,
                                               runningResults['DLoss'] / runningResults['batchSize'],
                                               runningResults['GLoss'] / runningResults['batchSize'],
                                               runningResults['DScore'] / runningResults['batchSize'],
                                               runningResults['GScore'] / runningResults['batchSize']))
                gc.collect()

            self.netG.eval()

            # learning rate is decayed by a factor of 10 every half of total epochs
            if (epoch + 1) % (self.args.nEpochs / 2) == 0:
                for param_group in self.optimizerG.param_groups:
                    param_group['lr'] /= 10.0
                # logger.info('Learning rate decay: lr=%s', (optimizerG.param_groups[0]['lr']))

            return runningResults









