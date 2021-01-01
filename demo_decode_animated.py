import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import argparse
import stereogram as stgm

import torch
from networks import *


parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=str, default=r'./test_videos/shark.gif', metavar='str',
                    help='dir to video file')
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
                    help='image size at the input end of the network')
parser.add_argument('--checkpoint_dir', type=str, default=r'./checkpoints', metavar='str',
                    help='dir to save checkpoints (default: ./checkpoints)')
args = parser.parse_args()



synthesizer = stgm.Stereogram(CANVAS_HEIGHT=args.in_size)

os.makedirs(args.out_folder, exist_ok=True)

# define the network and load the checkpoint
print('loading best checkpoint...')
net_G = define_G(args).to(device)
checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_ckpt.pt'),
                        map_location=None if torch.cuda.is_available() else device)
net_G.load_state_dict(checkpoint['model_G_state_dict'])
net_G.eval()


cap = cv2.VideoCapture(args.in_file)
m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
org_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
org_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fname_input = os.path.join(args.out_folder, 'demo_input.mp4')
fname_pred = os.path.join(args.out_folder, 'demo_pred.mp4')
fname_cat = os.path.join(args.out_folder, 'demo_cat.mp4')
fname_transfer = os.path.join(args.out_folder, 'demo_transfer.mp4')
video_writer_input = utils.VideoWriter(fname=fname_input, h=org_h, w=org_w, frame_rate=15, layout='default', display=False)
video_writer_pred = utils.VideoWriter(fname=fname_pred, h=org_h, w=org_w, frame_rate=15, layout='default', display=False)
video_writer_cat = utils.VideoWriter(fname=fname_cat, h=org_h, w=org_w*2, frame_rate=15, layout='default', display=False)
video_writer_transfer = utils.VideoWriter(fname=fname_transfer, h=org_h, w=org_w, frame_rate=15, layout='transfer', display=False)

for i in range(m_frames):

    _, img_org = cap.read()
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    org_h, org_w, c = img_org.shape
    img_ = cv2.resize(img_org, (args.in_size, args.in_size), cv2.INTER_CUBIC)

    img = torch.tensor(img_).permute([2, 0, 1]).unsqueeze(0).to(device)
    G_pred = net_G(img)
    out = G_pred.detach().squeeze().cpu().numpy()
    out = np.clip(out, a_min=0, a_max=1.0)
    out = utils.normalize(out, p_min=0.02, p_max=0.02)

    img_ = (img_org * 255).astype(np.uint8)
    out = (out * 255).astype(np.uint8)
    out = cv2.resize(out, (org_w, org_h), cv2.INTER_CUBIC)

    cm = plt.get_cmap('plasma')
    # Apply the colormap like a function to any array:
    out = (cm(out)*255.).astype(np.uint8)[:,:,0:3]

    cat = np.concatenate([img_, out], axis=1)

    video_writer_input.write_frame(img_)
    video_writer_pred.write_frame(out)
    video_writer_cat.write_frame(cat)
    video_writer_transfer.write_frame(img_before=img_, img_after=out, idx=i)

    print('processing %d / %d frames' % (i, m_frames))

