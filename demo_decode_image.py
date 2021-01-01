import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import argparse
import utils

import torch
from networks import *
import stereogram as stgm

parser = argparse.ArgumentParser()
parser.add_argument('--in_folder', type=str, default=r'./test_images', metavar='str',
                    help='input folder dir')
parser.add_argument('--out_folder', type=str, default=r'./decode_output', metavar='str',
                    help='output folder to save decoding results')
parser.add_argument('--net_G', type=str, default='unet_256', metavar='str',
                    help='net_G: resnet18 or resnet50 or unet_64 or unet_128 or unet_256 (default: resnet18)')
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
args = parser.parse_args()



img_dirs = os.listdir(args.in_folder)
os.makedirs(args.out_folder, exist_ok=True)

synthesizer = stgm.Stereogram(CANVAS_HEIGHT=args.in_size)

# define the network and load the checkpoint
print('loading best checkpoint...')
net_G = define_G(args).to(device)
checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_ckpt.pt'),
                        map_location=None if torch.cuda.is_available() else device)
net_G.load_state_dict(checkpoint['model_G_state_dict'])
net_G.eval()


m_imgs = len(img_dirs)
for i in range(m_imgs):
    this_img_path = os.path.join(args.in_folder, img_dirs[i])
    img_org = cv2.imread(this_img_path, cv2.IMREAD_COLOR)
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    org_h, org_w, c = img_org.shape
    img_ = cv2.resize(img_org, (args.in_size, args.in_size), cv2.INTER_CUBIC)

    img = torch.tensor(img_).permute([2, 0, 1]).unsqueeze(0).to(device)
    G_pred = net_G(img)
    out = G_pred.detach().squeeze().cpu().numpy()
    out = np.clip(out, a_min=0, a_max=1.0)
    out = utils.normalize(out, p_min=0.02, p_max=0.02)

    img_ = (img_org*255).astype(np.uint8)
    out = (out*255).astype(np.uint8)
    out = cv2.resize(out, (org_w, org_h), cv2.INTER_CUBIC)

    cv2.imwrite(os.path.join(args.out_folder, img_dirs[i]), img_[:,:,::-1])
    cv2.imwrite(os.path.join(args.out_folder, img_dirs[i].replace('.', '_pred.')), out)
    plt.imsave(os.path.join(args.out_folder, img_dirs[i].replace('.', '_pred_color.')), out, cmap='plasma')

    print('processing %d / %d images' % (i, m_imgs))

