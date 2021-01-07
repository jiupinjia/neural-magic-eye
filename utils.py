import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from skimage import measure

import torch
from torchvision import utils


def make_numpy_grid(tensor_data):

    # tensor_data: b x c x h x w, [0, 1], tensor
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)

    return vis


def cpt_ssim(img, img_gt, normalize=False):

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-9)

    SSIM = measure.compare_ssim(img, img_gt, data_range=1.0)

    return SSIM


def cpt_psnr(img, img_gt, PIXEL_MAX=1.0, normalize=False):

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-9)

    mse = np.mean((img - img_gt) ** 2)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    return psnr


def cpt_cos_similarity(img, img_gt, normalize=False):

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min() + 1e-9)

    cos_dist = np.sum(img*img_gt) / np.sqrt(np.sum(img**2)*np.sum(img_gt**2) + 1e-9)

    return cos_dist


def cpt_batch_psnr(img, img_gt, PIXEL_MAX):

    mse = torch.mean((img - img_gt) ** 2)
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

    return psnr


def cpt_batch_classification_acc(predicted, target):

    # predicted: b x c, logits [-inf, +inf]
    pred_idx = torch.argmax(predicted, dim=1).int()
    pred_idx = torch.reshape(pred_idx, [-1])
    target = torch.reshape(target, [-1])

    return torch.mean((pred_idx.int()==target.int()).float())


def normalize(img, mask=None, p_min=0, p_max=0):

    # img: h x w, [0, 1], np.float32
    if mask is None:
        sorted_arr = np.sort(img, axis=None)  # sort the flattened array
    else:
        sorted_arr = np.sort(img[mask == 1], axis=None)  # sort the flattened array
    n = len(sorted_arr)
    img_min = sorted_arr[int(n*p_min)]
    img_max = sorted_arr[::-1][int(n*p_max)]
    img_norm = (img - img_min) / (img_max - img_min + 1e-6)

    return np.clip(img_norm, a_min=0, a_max=1.0)



def get_sub_pxl_values(img, ys, xs):

    # img: h x w x c, [0, 1], np.float32
    h, w, c = img.shape
    xs0, ys0, xs1, ys1 = xs.astype(int), ys.astype(int), xs.astype(int) + 1, ys.astype(int) + 1
    xs1 = np.clip(xs1, a_min=0, a_max=w - 1)
    ys1 = np.clip(ys1, a_min=0, a_max=h - 1)

    dx = (xs - xs0).astype(np.float32)
    dy = (ys - ys0).astype(np.float32)
    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx) * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx) * (dy)

    weight_tl = np.expand_dims(weight_tl, axis=-1)
    weight_tr = np.expand_dims(weight_tr, axis=-1)
    weight_bl = np.expand_dims(weight_bl, axis=-1)
    weight_br = np.expand_dims(weight_br, axis=-1)

    pxl_values = weight_tl * img[ys0, xs0, :] + \
                 weight_tr * img[ys0, xs1, :] + \
                 weight_bl * img[ys1, xs0, :] + \
                 weight_br * img[ys1, xs1, :]

    return pxl_values



class VideoWriter:

    def __init__(self, fname='./demo.mp4',
                 h=760, w=1280,
                 frame_rate=10, bottom_crop=False,
                 layout='default', display=True):

        self.w = int(w)
        self.h = int(h)
        self.bottom_crop = bottom_crop
        self.layout = layout
        self.display = display
        self.bottom_crop = bottom_crop

        self.video_writer = cv2.VideoWriter(
            fname, cv2.VideoWriter_fourcc(*'MP4V'), frame_rate,
            (self.w, self.h))

    def write_frame(self, img_after, img_before=None, idx=None):

        if img_after.shape[0] != self.h or img_after.shape[1] != self.w:
            img_after = cv2.resize(img_after, (self.w, self.h))
            if img_before is not None:
                img_before = cv2.resize(img_before, (self.w, self.h))

        if self.layout == 'default':
            img = img_after
        if self.layout == 'transfer':
            img = np.zeros_like(img_after)
            start_frame_id, end_frame_dx = 20, 40
            s = int((idx - start_frame_id) / (end_frame_dx - start_frame_id) * self.w)
            s = np.clip(s, a_min=0, a_max=self.w)
            img[:, 0:s, :] = img_after[:, 0:s, :]
            img[:, s:, :] = img_before[:, s:, :]

        frame = img[:,:,::-1]

        if self.bottom_crop:
            h_crop = int(self.h * 0.9)
            frame = cv2.resize(frame[:h_crop, :, :], (self.w, self.h))

        self.video_writer.write(frame)

        if self.display:
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

