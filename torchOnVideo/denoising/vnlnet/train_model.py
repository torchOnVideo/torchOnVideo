import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os

from .vnlnet import VNLNet
from ..models import ModifiedDnCNN
from .utils import weights_init_kaiming, batch_PSNR

class TrainModel(VNLNet):
    def __init__(self):
        super(TrainModel, self).__init__()


    def __call__(self, *args, **kwargs):

        #### Shardul TODO Manage dataset
        print('> Loading dataset ...')
        dataset_train = Dataset(args.train_dir, color_mode=args.color, sigma=args.sigma,
                                oracle_mode=args.oracle_mode, past_frames=args.past_frames,
                                future_frames=args.future_frames,
                                search_window_width=args.search_window_width, nn_patch_width=args.nn_patch_width,
                                pass_nn_value=args.pass_nn_value)
        dataset_val = Dataset(args.val_dir, color_mode=args.color, sigma=args.sigma,
                              oracle_mode=args.oracle_mode, past_frames=args.past_frames,
                              future_frames=args.future_frames,
                              search_window_width=args.search_window_width, nn_patch_width=args.nn_patch_width,
                              pass_nn_value=args.pass_nn_value, patch_stride=20)
        loader_train = DataLoader(dataset=dataset_train, num_workers=2, \
                                  batch_size=args.batch_size, shuffle=True)
        loader_val = DataLoader(dataset=dataset_val, num_workers=2, \
                                batch_size=args.batch_size, shuffle=False)
        print('\t# of training samples: %d\n' % int(len(dataset_train)))

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        writer = SummaryWriter(args.save_dir)
        ######

        # Create model
        args.input_channels = dataset_train.data_num_channels()
        args.output_channels = (3 if args.color else 1)
        args.nlconv_features = (96 if args.color else 32)
        args.nlconv_layers = 4
        args.dnnconv_features = (192 if args.color else 64)
        args.dnnconv_layers = 15
        model = ModifiedDnCNN(input_channels=args.input_channels,
                              output_channels=args.output_channels,
                              nlconv_features=args.nlconv_features,
                              nlconv_layers=args.nlconv_layers,
                              dnnconv_features=args.dnnconv_features,
                              dnnconv_layers=args.dnnconv_layers)

        model.apply(weights_init_kaiming)
        criterion = nn.MSELoss(size_average=False)

        # Move to GPU
        device = torch.device("cuda:0")
        model.to(device)
        criterion.cuda()

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        step = 0
        current_lr = args.lr

        # Training
        for epoch in range(0, args.epochs):
            if (epoch + 1) >= args.milestone[1]:
                current_lr = args.lr / 1000.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            elif (epoch + 1) >= args.milestone[0]:
                current_lr = args.lr / 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            # train over all data in the epoch
            for i, data in enumerate(loader_train, 0):
                # Pre-training step
                model.train()
                model.zero_grad()
                optimizer.zero_grad()

                (stack_train, expected_train) = data

                stack_train = Variable(stack_train.cuda(), volatile=True)
                expected_train = Variable(expected_train.cuda(), volatile=True)

                # Evaluate model and optimize it
                out_train = model(stack_train)
                loss = criterion(out_train, expected_train) / (expected_train.size()[0] * 2)
                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    # Results
                    model.eval()
                    psnr_train = batch_PSNR(expected_train, out_train, 1.)
                    print('[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f' % \
                          (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
                    # Log the scalar values
                    writer.add_scalar('loss', loss.item(), step)
                    writer.add_scalar('PSNR on training data', psnr_train, step)
                step += 1

            model.eval()
            psnr_val = 0
            with torch.no_grad():
                for i, data in enumerate(loader_val, 0):
                    (stack_val, expected_val) = data
                    stack_val = Variable(stack_val.cuda())
                    expected_val = Variable(expected_val.cuda())
                    out_val = model(stack_val)
                    psnr_val += batch_PSNR(out_val, expected_val, 1.)
            psnr_val /= len(dataset_val)
            psnr_val *= args.batch_size

            print('\n[epoch %d] PSNR_val: %.4f' % (epoch + 1, psnr_val))
            writer.add_scalar('PSNR on validation data', psnr_val, epoch)
            writer.add_scalar('Learning rate', current_lr, epoch)

            net_data = { \
                'model_state_dict': model.state_dict(), \
                'args': args \
                }
            torch.save(net_data, os.path.join(args.save_dir, 'net.pth'))

            # Prepare next epoch
            dataset_train.prepare_epoch()
