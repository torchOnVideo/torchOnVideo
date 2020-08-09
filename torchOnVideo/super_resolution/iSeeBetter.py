import argparse
from tqdm import tqdm
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

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

        train_set = get_training_set(args.data_dir, args.nFrames, args.upscale_factor, args.data_augmentation,
                                     args.file_list,
                                     args.other_dataset, args.patch_size, args.future_frame)
        training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize,
                                          shuffle=True)





