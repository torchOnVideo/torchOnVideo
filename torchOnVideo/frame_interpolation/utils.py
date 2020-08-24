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