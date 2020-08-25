import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from torchOnVideo.super_resolution.utils import rgb2y, random_crop, augmentation_SOFVSR

class TrainSOFVSR(Dataset):
    def __init__(self, trainset_dir, scale, patch_size, n_iters, batch_size, degradation):
        super(TrainSOFVSR).__init__()
        self.trainset_dir = trainset_dir
        self.scale = scale
        self.patch_size = patch_size
        self.n_iters = n_iters * batch_size
        self.video_list = os.listdir(trainset_dir)
        self.degradation = degradation

    def __getitem__(self, idx):
        idx_video = random.randint(0, len(self.video_list)-1)
        idx_frame = random.randint(0, 28)                           # #frames of training videos is 31, 31-3=28
        lr_dir = self.trainset_dir + '/' + self.video_list[idx_video] + '/lr_x' + str(self.scale) + '_' + self.degradation
        hr_dir = self.trainset_dir + '/' + self.video_list[idx_video] + '/hr'

        # read HR & LR frames
        LR0 = Image.open(lr_dir + '/lr' + str(idx_frame) + '.png')
        LR1 = Image.open(lr_dir + '/lr' + str(idx_frame + 1) + '.png')
        LR2 = Image.open(lr_dir + '/lr' + str(idx_frame + 2) + '.png')
        HR0 = Image.open(hr_dir + '/hr' + str(idx_frame) + '.png')
        HR1 = Image.open(hr_dir + '/hr' + str(idx_frame + 1) + '.png')
        HR2 = Image.open(hr_dir + '/hr' + str(idx_frame + 2) + '.png')

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        HR0 = np.array(HR0, dtype=np.float32) / 255.0
        HR1 = np.array(HR1, dtype=np.float32) / 255.0
        HR2 = np.array(HR2, dtype=np.float32) / 255.0

        # extract Y channel for LR inputs
        HR0 = rgb2y(HR0)
        HR1 = rgb2y(HR1)
        HR2 = rgb2y(HR2)
        LR0 = rgb2y(LR0)
        LR1 = rgb2y(LR1)
        LR2 = rgb2y(LR2)

        # crop patchs randomly
        HR0, HR1, HR2, LR0, LR1, LR2 = random_crop(HR0, HR1, HR2, LR0, LR1, LR2, self.patch_size, self.scale)

        HR0 = HR0[:, :, np.newaxis]
        HR1 = HR1[:, :, np.newaxis]
        HR2 = HR2[:, :, np.newaxis]
        LR0 = LR0[:, :, np.newaxis]
        LR1 = LR1[:, :, np.newaxis]
        LR2 = LR2[:, :, np.newaxis]

        HR = np.concatenate((HR0, HR1, HR2), axis=2)
        LR = np.concatenate((LR0, LR1, LR2), axis=2)

        # data augmentation
        LR, HR = augmentation_SOFVSR()(LR, HR)

        LR_tensor = torch.from_numpy(LR.transpose((2, 0, 1)))
        HR_tensor = torch.from_numpy(HR.transpose((2, 0, 1)))
        return LR_tensor, HR_tensor

    def __len__(self):
        return self.n_iters

