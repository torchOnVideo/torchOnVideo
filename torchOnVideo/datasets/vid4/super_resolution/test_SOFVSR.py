from torch.utils.data import Dataset


class TestSOFVSR(Dataset):
    def __init__(self, cfg, video_name):
        super(TestSOFVSR).__init__()
        self.dataset_dir = cfg.testset_dir + '/' + video_name
        self.degradation = cfg.degradation
        self.scale = cfg.scale
        self.frame_list = os.listdir(self.dataset_dir + '/lr_x' + str(self.scale) + '_' + self.degradation)

    def __getitem__(self, idx):
        dir = self.dataset_dir + '/lr_x' + str(self.scale) + '_' + self.degradation
        LR0 = Image.open(dir + '/' + 'lr_' + str(idx+1).rjust(2, '0') + '.png')
        LR1 = Image.open(dir + '/' + 'lr_' + str(idx+2).rjust(2, '0') + '.png')
        LR2 = Image.open(dir + '/' + 'lr_' + str(idx+3).rjust(2, '0') + '.png')
        W, H = LR1.size

        # H and W should be divisible by 2
        W = int(W // 2) * 2
        H = int(H // 2) * 2
        LR0 = LR0.crop([0, 0, W, H])
        LR1 = LR1.crop([0, 0, W, H])
        LR2 = LR2.crop([0, 0, W, H])

        LR1_bicubic = LR1.resize((W*self.scale, H*self.scale), Image.BICUBIC)
        LR1_bicubic = np.array(LR1_bicubic, dtype=np.float32) / 255.0

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0

        # extract Y channel for LR inputs
        LR0_y, _, _ = rgb2ycbcr(LR0)
        LR1_y, _, _ = rgb2ycbcr(LR1)
        LR2_y, _, _ = rgb2ycbcr(LR2)

        LR0_y = LR0_y[:, :, np.newaxis]
        LR1_y = LR1_y[:, :, np.newaxis]
        LR2_y = LR2_y[:, :, np.newaxis]
        LR = np.concatenate((LR0_y, LR1_y, LR2_y), axis=2)

        LR = toTensor(LR)

        # generate Cr, Cb channels using bicubic interpolation
        _, SR_cb, SR_cr = rgb2ycbcr(LR1_bicubic)

        return LR, SR_cb, SR_cr

    def __len__(self):
        return len(self.frame_list) - 2

