from .fastdvdnet import  FastDVDNet

from ..models.fastdvdnet import FastDVDnet as FastDVDNet_model

# To be implemented
class TrainModel(FastDVDNet):
    def __init__(self):
        super(TrainModel, self).__init__()

    def __call__(self, *args, **kwargs):
        pass




