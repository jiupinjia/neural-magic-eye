import argparse
import numpy as np
import matplotlib.pyplot as plt
import datasets
from neural_decoder import *

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', metavar='str',
                    help='dataset name from [mnist, shapenet, watermarking, watermarking] (default: mnist)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 2)')
parser.add_argument('--net_G', type=str, default='unet_256', metavar='str',
                    help='net_G: resnet18fcn or resnet50fcn or unet_64 or unet_128 or unet_256 (default: resnet18)')
parser.add_argument('--norm_type', type=str, default='batch', metavar='str',
                    help='norm_type: instance or batch or none (default: batch)')
parser.add_argument('--with_disparity_conv', action='store_true', default=False,
                    help='insert a disparity convolution layer at the input end of the network')
parser.add_argument('--with_skip_connection', action='store_true', default=False,
                    help='using unet-fashion skip-connection at prediction layers')
parser.add_argument('--in_size', type=int, default=256, metavar='N',
                    help='input image size for training (default: 128)')
parser.add_argument('--checkpoint_dir', type=str, default=r'./checkpoints', metavar='str',
                    help='dir to save checkpoints (default: ./checkpoints)')
parser.add_argument('--vis_dir', type=str, default=r'./val_out', metavar='str',
                    help='dir to save results during training (default: ./val_out)')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_num_epochs', type=int, default=100, metavar='N',
                    help='max number of training epochs (default 200)')
parser.add_argument('--scheduler_step_size', type=int, default=50, metavar='N',
                    help='after m epochs then reduce lr to 0.1*lr (default 500)')
args = parser.parse_args()


if __name__ == '__main__':

    # # How to check if the data is loading correctly?
    # dataloaders = datasets.get_loaders(args)
    # for i in range(100):
    #     data = next(iter(dataloaders['train']))
    #     vis_A = utils.make_numpy_grid(data['stereogram'])
    #     vis_B = utils.make_numpy_grid(data['dmap'])
    #     vis = np.concatenate([vis_A, vis_B], axis=0)
    #     plt.imshow(vis)
    #     plt.show()

    dataloaders = datasets.get_loaders(args)
    nn_decoder = Decoder(args=args, dataloaders=dataloaders)
    nn_decoder.train_models()


