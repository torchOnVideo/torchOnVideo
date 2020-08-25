import torch
import os

from .adacof import AdaCoF
from ..models import AdaCoFNet
from torchOnVideo.losses import AdaCoF_loss
from torchOnVideo.datasets.Middlebury.frame_interpolation import TestAdaCoF

class TestModel(AdaCoF):
    def __init__(self, checkpoint=None, testset=TestAdaCoF, out_dir="../db/frame_interpolation/adacof/middlebury",
                 input_dir="../db/frame_interpolation/adacof/middlebury/input",
                 gt_dir="../db/frame_interpolation/adacof/middlebury/gt",
                 kernel_size=5, dilation=1, gpu_id=0, output_name='frame10i11.png'):
        super(TestModel, self).__init__()
        torch.cuda.set_device(gpu_id)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.test_set = testset

        self.model = AdaCoFNet(self.kernel_size, self.dilation)

        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.model.load(checkpoint['state_dict'])
        self.out_dir = out_dir

        self.current_epoch = checkpoint['epoch']

        self.output_name = output_name

    def __call__(self, *args, **kwargs):

        test_dir = self.out_dir
        test_db = self.test_set(self.input_dir, self.gt_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        test_db.Test(self.model, test_dir, self.current_epoch, output_name=self.output_name)


