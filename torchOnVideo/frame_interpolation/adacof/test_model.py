import torch
import os

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='adacofnet')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--config', type=str, default='./checkpoint/kernelsize_5/config.txt')
parser.add_argument('--out_dir', type=str, default='./output_adacof_test')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

from .adacof import AdaCoF
from ..models import AdaCoFNet
from torchOnVideo.losses import AdaCoF_loss
from torchOnVideo.datasets.Middlebury.frame_interpolation import TestAdaCoF

class TestModel(AdaCoF):
    def __init__(self, checkpoint=None, testset=TestAdaCoF, out_dir="../db/frame_interpolation/adacof/middlebury",
                 input_dir="../db/frame_interpolation/adacof/middlebury/input",
                 gt_dir="../db/frame_interpolation/adacof/middlebury/gt",
                 kernel_size=5, dilation=1, gpu_id=0, output_name='frame10i11.png'):
        torch.cuda.set_device(gpu_id)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.input_dir = input_dir
        self.gt_dir = gt_dir

        self.model = AdaCoFNet(self.kernel_size, self.dilation)

        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.model.load(checkpoint['state_dict'])
        self.out_dir = out_dir

        self.current_epoch = checkpoint['epoch']

        self.output_name = output_name


    def __call__(self, *args, **kwargs):

        test_dir = self.out_dir
        test_db = TestAdaCoF(self.input_dir, self.gt_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        test_db.Test(self.model, test_dir, self.current_epoch, output_name=self.output_name)


