import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random

import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import pyflow
from skimage import img_as_float
from random import randrange
import os.path

# From SOF VSR
def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    x = x.view(b, groups, c // groups, h, w)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(b, -1, h, w)
    return x


# From SOF VSR
def optical_flow_warp(image, image_optical_flow):
    """
    Arguments
        image_ref: reference images tensor, (b, c, h, w)
        image_optical_flow: optical flow to image_ref (b, 2, h, w)
    """
    b, _, h, w = image.size()
    grid = np.meshgrid(range(w), range(h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
    grid = grid.transpose(2, 0, 1)
    grid = np.tile(grid, (b, 1, 1, 1))
    grid = Variable(torch.Tensor(grid))
    if image_optical_flow.is_cuda == True:
        grid = grid.cuda()

    flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :] * 31 / (w - 1), dim=1)
    flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] * 31 / (h - 1), dim=1)
    grid = grid + torch.cat((flow_0, flow_1), 1)
    grid = grid.transpose(1, 2)
    grid = grid.transpose(3, 2)
    output = F.grid_sample(image, grid, padding_mode='border')
    return output


def L1_regularization(image):
    b, _, h, w = image.size()
    reg_x_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 1:, 0:w-1]
    reg_y_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 0:h-1, 1:]
    reg_L1 = torch.abs(reg_x_1) + torch.abs(reg_y_1)
    return torch.sum(reg_L1) / (b*(h-1)*(w-1))


def random_crop(HR0, HR1, HR2, LR0, LR1, LR2, patch_size_lr, scale):
    h_hr, w_hr = HR0.shape
    h_lr = h_hr // scale
    w_lr = w_hr // scale
    idx_h = random.randint(10, h_lr - patch_size_lr - 10)
    idx_w = random.randint(10, w_lr - patch_size_lr - 10)

    h_start_hr = (idx_h - 1) * scale
    h_end_hr = (idx_h - 1 + patch_size_lr) * scale
    w_start_hr = (idx_w - 1) * scale
    w_end_hr = (idx_w - 1 + patch_size_lr) * scale

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr

    HR0 = HR0[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR1 = HR1[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    HR2 = HR2[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    LR0 = LR0[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR1 = LR1[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    LR2 = LR2[h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    return HR0, HR1, HR2, LR0, LR1, LR2


def rgb2ycbcr(img_rgb):
    ## the range of img_rgb should be (0, 1)
    img_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    img_cb = -0.148 * img_rgb[:, :, 0] - 0.291 * img_rgb[:, :, 1] + 0.439 * img_rgb[:, :, 2] + 128 / 255.0
    img_cr = 0.439 * img_rgb[:, :, 0] - 0.368 * img_rgb[:, :, 1] - 0.071 * img_rgb[:, :, 2] + 128 / 255.0
    return img_y, img_cb, img_cr


def ycbcr2rgb(img_ycbcr):
    ## the range of img_ycbcr should be (0, 1)
    img_r = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 1.596 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_g = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) - 0.392 * (img_ycbcr[:, :, 1] - 128 / 255.0) - 0.813 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_b = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 2.017 * (img_ycbcr[:, :, 1] - 128 / 255.0)
    img_r = img_r[:, :, np.newaxis]
    img_g = img_g[:, :, np.newaxis]
    img_b = img_b[:, :, np.newaxis]
    img_rgb = np.concatenate((img_r, img_g, img_b), 2)
    return img_rgb


def rgb2y(img_rgb):
    ## the range of img_rgb should be (0, 1)
    image_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] +16 / 255.0
    return image_y


class augmentation_SOFVSR(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random()<0.5:
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random()<0.5:
            input = input.transpose(1, 0, 2)
            target = target.transpose(1, 0, 2)
        return np.ascontiguousarray(input), np.ascontiguousarray(target)


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    # random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)

        char_len = len(filepath)
        neigbor = []

        for i in seq:
            index = int(filepath[char_len - 7:char_len - 4]) - i
            file_name = filepath[0:char_len - 7] + '{0:03d}'.format(index) + '.png'

            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len - 7] + '{0:03d}'.format(index) + '.png').convert('RGB'),
                               scale).resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                # print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath, 'im' + str(nFrames) + '.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath + '/im' + str(j) + '.png').convert('RGB'), scale).resize(
            (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC) for j in reversed(seq)]

    return target, input, neigbor


def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames / 2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)

        char_len = len(filepath)
        neigbor = []
        if nFrames % 2 == 0:
            seq = [x for x in range(-tt, tt) if x != 0]  # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt, tt + 1) if x != 0]
        # random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len - 7:char_len - 4]) + i
            file_name1 = filepath[0:char_len - 7] + '{0:03d}'.format(index1) + '.png'

            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize(
                    (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                # print('neigbor frame- is not exist')
                temp = input
                neigbor.append(temp)

    else:
        target = modcrop(Image.open(join(filepath, 'im4.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4 - tt, 5 + tt) if x != 4]
        # random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath + '/im' + str(j) + '.png').convert('RGB'), scale).resize(
                (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC))
    return target, input, neigbor


def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                                         nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    # flow = rescale_flow(flow,0,1)
    return flow


def rescale_flow(x, max_range, min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range - min_range) / (max_val - min_val) * (x - max_val) + max_range


def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))
    return img


def get_patch(img_in, img_tar, img_nn, patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))  # [:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_nn]  # [:, iy:iy + ip, ix:ix + ip]

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, info_patch


def augment(img_in, img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, info_aug


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in
