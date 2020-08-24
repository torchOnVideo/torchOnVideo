import torch
from torch.autograd import Variable


def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data ** 2 + epsilon ** 2))


def moduleNormalize(frame):
    return torch.cat([(frame[:, 0:1, :, :] - 0.4631), (frame[:, 1:2, :, :] - 0.4352), (frame[:, 2:3, :, :] - 0.3990)],
                     1)
# Shardul Check
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def build_input_CAIN(images, imgpaths, is_training=True, include_edge=False, device=torch.device('cuda')):
    if isinstance(images[0], list):
        images_gathered = [None, None, None]
        for j in range(len(images[0])):  # 3
            _images = [images[k][j] for k in range(len(images))]
            images_gathered[j] = torch.cat(_images, 0)
        imgpaths = [p for _ in images for p in imgpaths]
        images = images_gathered

    im1, im2 = images[0].to(device), images[2].to(device)
    gt = images[1].to(device)

    return im1, im2, gt
