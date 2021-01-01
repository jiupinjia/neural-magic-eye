import argparse
import numpy as np
import matplotlib.pyplot as plt
import datasets
from neural_decoder import *

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', metavar='str',
                    help='dataset name from [mnist, shapenet, watermarking] (default: mnist)')
parser.add_argument('--prediction_mode', type=str, default='stereogram2label', metavar='str',
                    help='dataset name from [stereogram2label, depth2label] (default: mnist)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--net_G', type=str, default='resnet18', metavar='str',
                    help='net_G: resnet18 or resnet50 (default: resnet18)')
parser.add_argument('--norm_type', type=str, default='batch', metavar='str',
                    help='norm_type: instance or batch or none (default: batch)')
parser.add_argument('--with_disparity_conv', action='store_true', default=False,
                    help='insert a disparity convolution layer at the input end of the network')
parser.add_argument('--in_size', type=int, default=64, metavar='N',
                    help='input image size for training [64, 128, 256]  (default: 64)')
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

    dataloaders = datasets.get_loaders(args)
    nn_classifier = Classifier(args=args, dataloaders=dataloaders)
    nn_classifier.train_models()

