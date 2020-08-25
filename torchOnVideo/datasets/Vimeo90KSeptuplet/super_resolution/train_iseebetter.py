from torch.utils.data import Dataset
import os
from os.path import join
import torchvision.transforms as transforms
import random
from PIL import Image

from torchOnVideo.super_resolution.utils import *

class TrainISeeBetter(Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,
                 future_frame, transform=None):
        super(TrainISeeBetter, self).__init__()

        alist = [line.rstrip() for line in open(join(image_dir, file_list))]
        self.image_filenames = [join(image_dir, x) for x in alist]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame

    def __getitem__(self, index):
        if self.future_frame:
            target, input, neigbor = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                                     self.other_dataset)
        else:
            target, input, neigbor = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor,
                                              self.other_dataset)

        if self.patch_size != 0:
            input, target, neigbor, _ = get_patch(input, target, neigbor, self.patch_size, self.upscale_factor,
                                                  self.nFrames)

        if self.data_augmentation:
            input, target, neigbor, _ = augment(input, target, neigbor)

        bicubic = rescale_img(input, self.upscale_factor)

        # Shardul change issue in get_flow
        flow = [get_flow(input, j) for j in neigbor]

        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            flow = [torch.from_numpy(j.transpose(2, 0, 1)) for j in flow]

        return input, target, neigbor, flow, bicubic

    def __len__(self):
        return len(self.image_filenames)
