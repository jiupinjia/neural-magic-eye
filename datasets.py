import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
import glob
import random

import stereogram as stgm
import utils

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import MNIST

import string

import qrcode

class DataAugmentation:

    def __init__(self,
                 with_random_hflip=False,
                 with_random_vflip=False,
                 with_random_blur=False,
                 with_random_rotate=False,
                 with_random_crop=False,
                 with_random_aspect_ratio=False,
                 with_random_jpeg_compression=False):

        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_blur = with_random_blur
        self.with_random_rotate = with_random_rotate
        self.with_random_crop = with_random_crop
        self.with_random_aspect_ratio = with_random_aspect_ratio
        self.with_random_jpeg_compression = with_random_jpeg_compression

    def transform(self, img):

        h, w = img.shape[0:2]

        if self.with_random_hflip and random.random() > 0.5:
            img = img[:, ::-1]

        if self.with_random_vflip and random.random() > 0.5:
            img = img[::-1, :]

        if self.with_random_blur and random.random() > 0.5:
            k = random.randint(1, int(min(h, w)/20 + 1))
            img = cv2.blur(img, (k, k))

        if self.with_random_rotate and random.random() > 0.5:
            theta = random.uniform(-180, 180)
            image_center = tuple(np.array(img.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, theta, 1.0)
            img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        if self.with_random_crop and random.random() > 0.5:
            crop_h = random.uniform(h/2, h)
            crop_w = random.uniform(w/2, w)
            y1 = int(random.uniform(0, h-crop_h))
            y2 = int(y1 + crop_h)
            x1 = int(random.uniform(0, w-crop_w))
            x2 = int(x1 + crop_w)
            img = img[y1:y2, x1:x2]

        if self.with_random_aspect_ratio and random.random() > 0.5:
            target_aspect_ratio = random.uniform(3, 12)
            h_new = h
            w_new = int(h_new / target_aspect_ratio)
            img = cv2.resize(img, (w_new, h_new), cv2.INTER_CUBIC)

        if self.with_random_jpeg_compression and random.random() > 0.5:
            img = (img * 255.).astype(np.uint8)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(20, 90)]
            _, imgcode = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(imgcode, cv2.IMREAD_COLOR)
            img = img.astype(np.float32) / 255.

        return img


class SimpleMNISTDataset(Dataset):

    def __init__(self, bg_tile_dir, img_size, is_train=True):

        self.img_size = img_size
        self.synthesizer = stgm.Stereogram(CANVAS_HEIGHT=img_size)

        self.is_train = is_train
        _ = MNIST(root=r'./datasets', train=True, download=True)
        mnist_training_imgs, mnist_training_labels = torch.load(r'./datasets/MNIST/processed/training.pt')
        mnist_testing_imgs, mnist_testing_labels = torch.load(r'./datasets/MNIST/processed/test.pt')

        if is_train:
            self.bg_tiles_dirs = glob.glob(os.path.join(bg_tile_dir, 'train', '*.jpg'))
            self.mnist_imgs = np.array(mnist_training_imgs, dtype=np.float32) / 255.
            self.mnist_labels = np.array(mnist_training_labels, dtype=np.int)
            self.tile_augmenter = DataAugmentation(
                with_random_vflip=True, with_random_hflip=True, with_random_blur=True)
            self.dmap_augmenter = DataAugmentation(with_random_blur=True)
        else:
            self.bg_tiles_dirs = glob.glob(os.path.join(bg_tile_dir, 'val', '*.jpg'))
            self.mnist_imgs = np.array(mnist_testing_imgs, dtype=np.float32) / 255.
            self.mnist_labels = np.array(mnist_testing_labels, dtype=np.int)
            self.tile_augmenter = DataAugmentation()
            self.dmap_augmenter = DataAugmentation()

    def __len__(self):
        return len(self.mnist_labels)

    def __getitem__(self, idx):

        dmap = np.reshape(self.mnist_imgs[idx, :], [28, 28])
        label = self.mnist_labels[idx]

        idx = random.randint(0, len(self.bg_tiles_dirs) - 1)
        bg_tile = cv2.imread(self.bg_tiles_dirs[idx], cv2.IMREAD_COLOR)
        bg_tile = cv2.cvtColor(bg_tile, cv2.COLOR_BGR2RGB) / 255.

        bg_tile = self.tile_augmenter.transform(bg_tile)
        dmap = self.dmap_augmenter.transform(dmap)

        bg_tile, dmap = self.synthesizer.normalize_height(bg_tile, dmap)
        stereogram = self.synthesizer.synthesis(bg_tile, dmap)

        # resize and to tensor
        stereogram = cv2.resize(stereogram, (self.img_size, self.img_size), cv2.INTER_CUBIC)
        dmap = cv2.resize(dmap, (self.img_size, self.img_size), cv2.INTER_CUBIC)
        stereogram = TF.to_tensor(np.array(stereogram, dtype=np.float32))
        dmap = TF.to_tensor(np.array(dmap, dtype=np.float32))
        label = torch.tensor(label, dtype=torch.int)

        data = {'stereogram': stereogram, 'dmap': dmap, 'label': label}

        return data


class ShapeNetDataset(Dataset):

    def __init__(self, depth_map_dir, bg_tile_dir, img_size, is_train=True):

        self.img_size = img_size

        self.is_train = is_train
        if is_train:
            self.bg_tiles_dirs = glob.glob(os.path.join(bg_tile_dir, 'train', '*.jpg'))
            self.depth_map_files = np.loadtxt(os.path.join(depth_map_dir, 'train.txt'), dtype=np.str, delimiter=',')
            self.tile_augmenter = DataAugmentation(
                with_random_vflip=True, with_random_hflip=True,
                with_random_blur=True, with_random_aspect_ratio=True)
            self.dmap_augmenter = DataAugmentation(
                with_random_vflip=True, with_random_hflip=True,
                with_random_rotate=True, with_random_crop=True)
            self.stereogram_augmenter = DataAugmentation(with_random_jpeg_compression=True)
        else:
            self.bg_tiles_dirs = glob.glob(os.path.join(bg_tile_dir, 'val', '*.jpg'))
            self.depth_map_files = np.loadtxt(os.path.join(depth_map_dir, 'val.txt'), dtype=np.str, delimiter=',')
            self.tile_augmenter = DataAugmentation()
            self.dmap_augmenter = DataAugmentation()
            self.stereogram_augmenter = DataAugmentation()

        self.labels = self.depth_map_files[:, 2].astype(int)

    def __len__(self):
        return self.depth_map_files.shape[0]

    def __getitem__(self, idx):

        # why CANVAS_HEIGHT is set larger than in_size?
        # We want to simulate the degradation of image resize at inference time
        canvas_height = int(self.img_size*random.uniform(1.0, 1.5))
        synthesizer = stgm.Stereogram(CANVAS_HEIGHT=canvas_height)

        dmap = cv2.imread(self.depth_map_files[idx, 0], cv2.IMREAD_GRAYSCALE)
        dmap = dmap.astype(np.float32) / 255.
        label = self.labels[idx]

        idx = random.randint(0, len(self.bg_tiles_dirs) - 1)
        bg_tile = cv2.imread(self.bg_tiles_dirs[idx], cv2.IMREAD_COLOR)
        bg_tile = cv2.cvtColor(bg_tile, cv2.COLOR_BGR2RGB) / 255.

        bg_tile = self.tile_augmenter.transform(bg_tile)
        dmap = self.dmap_augmenter.transform(dmap)

        bg_tile, dmap = synthesizer.normalize_height(bg_tile, dmap)
        stereogram = synthesizer.synthesis(bg_tile, dmap)
        stereogram = self.stereogram_augmenter.transform(stereogram)

        # resize and to tensor
        stereogram = cv2.resize(stereogram, (self.img_size, self.img_size), cv2.INTER_CUBIC)
        dmap = cv2.resize(dmap, (self.img_size, self.img_size), cv2.INTER_CUBIC)
        stereogram = TF.to_tensor(np.array(stereogram, dtype=np.float32))
        dmap = TF.to_tensor(np.array(dmap, dtype=np.float32))
        label = torch.tensor(label, dtype=torch.int)

        data = {'stereogram': stereogram, 'dmap': dmap, 'label': label}

        return data



class WatermarkingDataset(Dataset):

    def __init__(self, base_canvas_dir, img_size, is_train=True):

        # we use fixed texture to generate autostereogram for both training and testing
        self.bg_tile = cv2.imread(r'./datasets/Textures/train/00099.jpg', cv2.IMREAD_COLOR)
        self.bg_tile = cv2.cvtColor(self.bg_tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.

        self.is_train = is_train
        if is_train:
            self.canvas_dir = glob.glob(os.path.join(base_canvas_dir, 'train', '*.jpg'))
            self.canvas_augmenter = DataAugmentation(with_random_hflip=True, with_random_crop=True, with_random_blur=True)
        else:
            self.canvas_dir = glob.glob(os.path.join(base_canvas_dir, 'val', '*.jpg'))
            self.canvas_augmenter = DataAugmentation()

        self.img_size = img_size

    def __len__(self):
        if self.is_train:
            return 50000
        else:
            return 5000

    def __getitem__(self, idx):

        # why CANVAS_HEIGHT is set larger than in_size?
        # We want to simulate the degradation of image resize at inference time
        canvas_height = int(self.img_size*random.uniform(1.0, 1.5))
        synthesizer = stgm.Stereogram(CANVAS_HEIGHT=canvas_height)

        idx = random.randint(0, len(self.canvas_dir) - 1)
        canvas = cv2.imread(self.canvas_dir[idx], cv2.IMREAD_COLOR)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        canvas = self.canvas_augmenter.transform(canvas)

        characters = string.ascii_letters
        length = random.randint(1, 50)
        random_str = ''.join([random.choice(characters) for j in range(length)])
        dmap = qrcode.make(random_str)
        dmap = 1 - np.array(dmap, np.float32)

        bg_tile, dmap = synthesizer.normalize_height(self.bg_tile, dmap)
        stereogram = synthesizer.synthesis(bg_tile, dmap)

        # resize
        stereogram = cv2.resize(stereogram, (self.img_size, self.img_size), cv2.INTER_CUBIC)
        dmap = cv2.resize(dmap, (self.img_size, self.img_size), cv2.INTER_CUBIC)
        canvas = cv2.resize(canvas, (self.img_size, self.img_size), cv2.INTER_CUBIC)

        alpha = random.uniform(0.1, 0.9)
        mix = alpha * stereogram + (1 - alpha) * canvas

        dmap = TF.to_tensor(np.array(dmap, dtype=np.float32))
        mix = TF.to_tensor(np.array(mix, dtype=np.float32))

        data = {'stereogram': mix, 'dmap': dmap}

        return data



def get_loaders(args):

    if args.dataset == 'mnist':
        training_set = SimpleMNISTDataset(
            bg_tile_dir=r'./datasets/Textures', img_size=args.in_size, is_train=True)
        val_set = SimpleMNISTDataset(
            bg_tile_dir=r'./datasets/Textures', img_size=args.in_size, is_train=False)
    elif args.dataset == 'shapenet':
        training_set = ShapeNetDataset(
            depth_map_dir=r'./datasets/ShapeNetCore.v2', bg_tile_dir=r'./datasets/Textures',
            img_size=args.in_size, is_train=True)
        val_set = ShapeNetDataset(
            depth_map_dir=r'./datasets/ShapeNetCore.v2', bg_tile_dir=r'./datasets/Textures',
            img_size=args.in_size, is_train=False)
    elif args.dataset == 'watermarking':
        training_set = WatermarkingDataset(
            base_canvas_dir=r'./datasets/VGGFlowers', img_size=args.in_size, is_train=True)
        val_set = WatermarkingDataset(
            base_canvas_dir=r'./datasets/VGGFlowers', img_size=args.in_size, is_train=False)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [maps, flowers, facades])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    return dataloaders

