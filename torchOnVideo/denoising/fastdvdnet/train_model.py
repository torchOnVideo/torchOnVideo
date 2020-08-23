from .fastdvdnet import  FastDVDNet

from ..models.fastdvdnet import FastDVDnet as FastDVDNet_model

class TrainModel(FastDVDNet):
    def __init__(self):
        super(TrainModel, self).__init__()

    def __call__(self, *args, **kwargs):

        #### Shardul TODO
        print('> Loading datasets ...')
        dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
        loader_train = train_dali_loader(batch_size=args['batch_size'], \
                                         file_root=args['trainset_dir'], \
                                         sequence_length=args['temp_patch_size'], \
                                         crop_size=args['patch_size'], \
                                         epoch_size=args['max_number_patches'], \
                                         random_shuffle=True, \
                                         temp_stride=3)

        num_minibatches = int(args['max_number_patches'] // args['batch_size'])
        ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
        print("\t# of training samples: %d\n" % int(args['max_number_patches']))

        # Init loggers
        writer, logger = init_logging(args)




