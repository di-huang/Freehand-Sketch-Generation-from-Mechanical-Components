import os
import numpy as np
import random
from PIL import Image
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle

from utils.sketch_utils import *
from utils.my_utils import *


class SketchDataset(Dataset):
    def __init__(self, args, sketch_root, data_root, transform, augment=False, test_only=False):
        self.paths_dict = {}
        self.augment = augment
        self.transform = transform
        self.label_data_source = args.label_data_source
        self.test_only = test_only

        if test_only:
            self.test_imgs = []
            for root, dirs, files in os.walk(data_root):
                for filename in files:
                    if not filename.lower().endswith(('.png', '.jpg')):
                        continue
                    filepath = os.path.join(root, filename)
                    self.test_imgs.append(filepath)
            return

        if self.label_data_source == 'init_clipasso':
            paths_dict_ = get_path_dict(args, data_root)
        else:
            sketch_dir = os.path.join(sketch_root, f"path.pkl")
            with open(sketch_dir, "rb") as f:
                paths_dict_ = pickle.load(f)

        for key, val in paths_dict_.items():        # key: [idx]_[seed]
            # discard if it does not contain information about both the initial stroke and L intermediate strokes.
            if len(val['iterations']) != 9:
                continue

            # change
            data_idx, seed = key.split("_")
            data_idx = int(data_idx)
            seed = int(seed)
            if data_idx in self.paths_dict:
                self.paths_dict[data_idx][seed] = val
            else:
                self.paths_dict[data_idx] = {seed: val}

    def __getitem__(self, index):
        if self.test_only:
            img_path = self.test_imgs[index]
            image = Image.open(img_path).convert('RGB')
            res_img = self.transform(image)
            return res_img, img_path

        # NOTE: seed set to be 0 by default
        path = self.paths_dict[index][0]['iterations']
        pos_list = []
        for idx in sorted(map(int, path.keys())):
            pos = torch.tensor(path[f"{idx}"]["pos"])
            pos_list.append(pos)
        res_pos = torch.stack(pos_list, dim=0)

        # NOTE: seed set to be 0 by default
        img_path = self.paths_dict[index][0]['img_path']
        image = Image.open(img_path).convert('RGB')
        res_img = self.transform(image)

        return res_img, res_pos

    def __len__(self):
        if self.test_only:
            return len(self.test_imgs)
        return len(self.paths_dict)


def get_dataset(args, test_only=False):
    sketch_root = os.path.join('logs', args.dataset)
    data_root = args.data_root
    test_root = os.path.join(args.data_root, 'test')
    image_shape = (3, args.custom_transforms_size, args.custom_transforms_size)

    transform = transforms.Compose([
        transforms.Resize((image_shape[1], image_shape[1])),
        transforms.ToTensor(),
    ])

    test_dataset = SketchDataset(
        args,
        None,
        test_root,
        transform,
        test_only=True
    )

    if test_only:
        return test_dataset, image_shape

    train_dataset = SketchDataset(
        args,
        sketch_root,
        data_root,
        transform,
    )

    if args.train_val_split_ratio > 0:
        train_dataset, val_dataset = split_dataset(train_dataset, args.train_val_split_ratio, seed=args.seed, use_stratify=False)
    else:
        val_dataset = train_dataset

    return train_dataset, val_dataset, test_dataset, image_shape

