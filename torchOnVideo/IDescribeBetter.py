from __future__ import print_function
from train import main,pretty_evaluation
from tensorboardX import SummaryWriter
import torch
import run
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import EvalConfig as Ev
from utils import evaluate, get_lr, load_checkpoint, save_checkpoint, test, train
from config import TrainConfig as C
from loader.MSVD import MSVD
from loader.MSRVTT import MSRVTT
from models.decoder import Decoder
from pytorch-video-feature-extractor import main as feat_ext
from models.caption_generator import CaptionGenerator
class Describe:

    def train(dataset="MSVD",lr=0.00001,ep=50):
        if dataset=="MSVD" and lr==0.00001 and ep == 50:
            print("training with default params")
            main(dataset="MSVD",lr=0.00001,ep=50)
        else:
            print("Your parameters are :")
            print("dataset : ",dataset)
            print("lr : ",lr)
            print("ep :",ep)
            main(dataset,lr,ep)


        pass

    def infer():
        print("Evaluates your model over MSVD dataset")
        run.run(Ev.ckpt_fpath)
        pass

    def Create_My_Corpus():
        print("Caution: Computationally expensive")
        feat_ext.pretty_extractor()
        pass

    def view_my_params():
        Print("below given are all the parameters and their default values")
        pass

    def use_trained():
        print("Loading the model from theis checkpoint")
        pass
