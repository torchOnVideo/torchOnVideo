from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
import random
from PIL import Image

# Shardul check remove the train, test mode later on
class TestCAIN(Dataset):
    def __init__(self, data_root, mode='hard'):
        '''
        :param data_root:   ./data/SNU-FILM
        :param mode:        ['easy', 'medium', 'hard', 'extreme']
        '''
        test_root = os.path.join(data_root, 'test')
        test_fn = os.path.join(data_root, 'test-%s.txt' % mode)
        with open(test_fn, 'r') as f:
            self.frame_list = f.read().splitlines()
        self.frame_list = [v.split(' ') for v in self.frame_list]

        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        print("[%s] Test dataset has %d triplets" % (mode, len(self.frame_list)))

    def __getitem__(self, index):
        # Use self.test_all_images:
        imgpaths = self.frame_list[index]

        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])
        img3 = Image.open(imgpaths[2])

        img1 = self.transforms(img1)
        img2 = self.transforms(img2)
        img3 = self.transforms(img3)

        imgs = [img1, img2, img3]

        return imgs, imgpaths

    def __len__(self):
        return len(self.frame_list)