from ..video_denoising import VideoDenoising

# parser = argparse.ArgumentParser(description='VNLnet Training')
# parser.add_argument('--train_dir', type=str, default='train/', help='Path containing the training data')
# parser.add_argument('--val_dir', type=str, default='val/', help='Path containing the validation data')
# parser.add_argument('--save_dir', type=str, default='mynetwork', help='Path to store the logs and the network')
# parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
# parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
# parser.add_argument('--milestone', nargs=2, type=int, default=[12, 17], help='When to decay learning rate')
# parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
# parser.add_argument('--sigma', type=float, default=20, help='Simulated noise level')
# parser.add_argument('--color', action='store_true', help='Train with color instead of grayscale')
# parser.add_argument('--oracle_mode', type=int, default=0, help='Oracle mode (0: no oracle, 1: image ground truth)')
# parser.add_argument('--past_frames', type=int, default=7, help='Number of past frames')
# parser.add_argument('--future_frames', type=int, default=7, help='Number of future frames')
# parser.add_argument('--search_window_width', type=int, default=41, help='Search window width for the matches')
# parser.add_argument('--nn_patch_width', type=int, default=41, help='Width of the patches for matching')
# parser.add_argument('--pass_nn_value', action='store_true', \
#                     help='Whether to pass the center pixel value of the matches (noisy image)')
# args = parser.parse_args()
#
# # The images are normalized
# args.sigma /= 255.


class VNLNet(VideoDenoising):
    def __init__(self):
        super(VNLNet, self).__init__()
