import os
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms.functional as functional
from sklearn.model_selection import train_test_split
import json

from models.LBS import SketchModel

from third_party.U2Net_.model import U2NET

import shlex

import argparser
from utils.config import Config
from utils.shared import update_args, update_config

U2Net = None


def mask_image(im, use_gpu=True):
    c, h, w = im.shape[1:]

    global U2Net

    if U2Net is None:
        U2Net = U2NET(3, 1)
        current_directory = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(os.path.dirname(current_directory), "third_party/U2Net_/saved_models/u2net.pth")
        if torch.cuda.is_available() and use_gpu:
            U2Net.load_state_dict(torch.load(ckpt_path))
            U2Net.cuda()
        else:
            U2Net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        U2Net.eval()

    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    )

    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = U2Net(data_transforms(im).detach())
    pred = d1[:, 0, :, :]

    pred_min = pred.flatten(1).min(dim=1)[0].view(-1, 1, 1)
    pred_max = pred.flatten(1).max(dim=1)[0].view(-1, 1, 1)
    predict = (pred - pred_min) / (pred_max - pred_min + 1e-6)

    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1

    mask = predict.unsqueeze(1).repeat(1, c, 1, 1)
    mask = transforms.functional.resize(mask, (h, w), transforms.InterpolationMode.NEAREST)

    return mask


def check_nan(tensor):
    if (tensor != tensor).sum() != 0:
        return True
    return False


def set_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """CosineAnnealingWarmUpRestarts sheduler."""
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1.0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up)))
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def count_parameters(model):
    """Return the number of parameters of the model ."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_dataset(dataset, split_ratio=0.2, seed=0, use_stratify=True):
    """Split dataset into two splits.

    Args:
        dataset (torch.utils.data.Dataset): dataset to be split
        split_ratio (float): split ratio. Defaults to 0.2.
        seed (int): random seed. Defaults to 0.
        use_stratify (bool): Data is split in a stratified fashion, using this as the class labels. Defaults to True.

    Returns:
        list: List contaning train-test split of the given dataset, with size [1-split_ratio, split_ratio].
    """
    if use_stratify:
        stratify = [dataset[i][1] for i in range(len(dataset))]
    else:
        stratify = None

    train_idx, valid_idx = train_test_split(
        np.arange(len(dataset)), test_size=split_ratio, shuffle=True, stratify=stratify, random_state=seed
    )

    train = torch.utils.data.Subset(dataset, train_idx)
    val = torch.utils.data.Subset(dataset, valid_idx)

    return train, val


class ImageGrid:
    def __init__(self, num_img=10, nrow=6):
        self.num_img = num_img * nrow
        self.nrow = nrow
        self._figures = []

    def update(self, *images):
        num_grid = len(images)
        fig, axs = plt.subplots(
            1,
            num_grid,
            figsize=(4 * num_grid, self.num_img // self.nrow / 2),
            constrained_layout=True,
            gridspec_kw={"wspace": 0.01},
        )
        for idx, img in enumerate(images):
            grid = make_grid(img[: self.num_img].detach().cpu(), nrow=self.nrow).permute(1, 2, 0).numpy()
            if num_grid == 1:
                axs.imshow(grid)
            else:
                axs[idx].imshow(grid)

        self._figures.append(fig)
        return fig

    def summary(self):
        return self._figures

    def reset(self):
        self._figures = {}


color_jitter_fn = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
weak_color_jitter_fn = transforms.ColorJitter(0.3, 0.2, 0.2, 0.05)


def sample_crop_box(min_crop_frac):
    l, r, t, b = 0, 0, 0, 0

    while r - l < min_crop_frac or b - t < min_crop_frac:
        sample = np.random.uniform(0, 1, 2)
        sample.sort()
        l, r = sample
        sample = np.random.uniform(0, 1, 2)
        sample.sort()
        t, b = sample

    return l, r, t, b


def random_aug(
    img, mask, gt_p, gt_c, n_back=0, 
    min_crop_frac=0.8, flip_p=0.5, jitter_p=0.8, jitter_weak=False
):
    """Apply identical random augmentation to image and corresponding mask, ground truth stroke (position and color).

    Args:
        img (torch.Tensor): Single target image, batch not supported.
        mask (torch.Tensor): Corresponding mask image
        p (torch.Tensor): Corresponding gt stroke parameter (position)
        c (torch.Tensor): Corresponding gt stroke parameter (color)
        n_back (torch.Tensor): # of background strokes in p & c. Defaults to 0.
        min_crop_frac (float): Minimum length proportion of cropped area when applying random cropping. Defaults to 0.8.
        flip_p (float): Probability of applying horizontal flipping. Defaults to 0.5.
        jitter_p (float): Probability of applying color jittering. Defaults to 0.8.
        jitter_weak (bool): Apply weak color jittering. Defaults to False.

    Returns:
        [type]: [description]
    """
    h, w = img.shape[1:]
    img = img.clone()
    mask = mask.clone()
    gt_p = gt_p.clone()
    gt_c = gt_c.clone()

    global color_jitter_fn
    global weak_color_jitter_fn

    jitter_fn = weak_color_jitter_fn if jitter_weak else color_jitter_fn

    ### sample cropping area and apply to img, mask
    l, r, t, b = sample_crop_box(min_crop_frac)

    loc = np.array([l * w, r * w, t * h, b * h]).astype(int)
    img = img[:, loc[2] : loc[3], loc[0] : loc[1]]
    img = functional.resize(img, (h, w))
    mask = mask[:, loc[2] : loc[3], loc[0] : loc[1]]
    mask = functional.resize(mask, (h, w))

    ### apply horizontal flipping
    if np.random.rand() < flip_p:
        img = img.flip(2)
        mask = mask.flip(2)
        l, r = r, l

    ### apply cropping into stroke parameters
    gt_p = gt_p.view(*gt_p.shape[:2], -1, 2) * 0.5 + 0.5
    gt_p[:, :, :, 0] = (gt_p[:, :, :, 0] - t) / (b - t) * 2 - 1
    gt_p[:, :, :, 1] = (gt_p[:, :, :, 1] - l) / (r - l) * 2 - 1
    ### reorder ground truth stroke parameters
    gt_p, gt_c = ordering_stroke(gt_p, gt_c, n_back)

    ### apply color jittering
    if np.random.rand() < jitter_p:
        nlayers, nlines = gt_c.shape[:2]
        color = gt_c.permute(2, 0, 1).repeat(1, h // nlayers + 1, 1)[:, :h, :]
        img = torch.cat([img, color], dim=2)
        img, color = jitter_fn(img).split([w, nlines], dim=2)
        gt_c = color[:, :nlayers, :].permute(1, 2, 0)

    return img, mask, gt_p, gt_c


def ordering_stroke(gt_p, gt_c=None, n_back=0):
    """Order the strokes parameters,
    For the direction of the control point for each stroke: left to right.
    For the index order between strokes: x-axis coordinates of the first control point of the initial position

    Args:
        p (torch.Tensor): [# of intermediate layers, # of strokes, # of control point (4), 2D coordinate (2)]
        c (torch.Tensor): [# of intermediate layers, # of strokes, # of channels (3)]
        n_back (int): # of background strokes. Defaults to 0.

    Returns:
        list: List of ordered stroke parameters (position and color)
    """
    ### stroke direction ordering
    if len(gt_p.shape) == 3:
        gt_p = gt_p.view(*gt_p.shape[:2], -1, 2)
    pos_order = (gt_p[:, :, 0, 0] < gt_p[:, :, -1, 0]).unsqueeze(-1).unsqueeze(-1)
    gt_p = torch.where(pos_order, gt_p, gt_p.flip(2)).flatten(2)

    ### storke index ordering
    idx_order = torch.sort(gt_p[0, :-n_back, 0], dim=0)[1]
    gt_p[:, :-n_back] = gt_p[:, idx_order]
    if n_back != 0:
        back_order = torch.sort(gt_p[0, -n_back:, 0], dim=0)[1]
        gt_p[:, -n_back:] = gt_p[:, -n_back:][:, back_order]
    gt_p = gt_p.clip(-1.2, 1.2)

    if gt_c is not None:
        gt_c[:, :-n_back] = gt_c[:, idx_order]
        if n_back != 0:
            gt_c[:, -n_back:] = gt_c[:, -n_back:][:, back_order]

    return gt_p, gt_c


def load_model(args):
    with open(os.path.join(args.path, "meta.json"), "r") as f:
        train_args = json.loads(f.read())["args"]

    train_args = Config.from_dict(train_args)
    stroke_config = argparser.get_stroke_config(train_args)
    update_args(train_args)
    update_config(stroke_config)
    model = SketchModel()
    model.to(train_args.device)
    checkpoint = torch.load(os.path.join(args.path, "model_best.pt"))
    model.load_state_dict(checkpoint)
    model.set_progress(1.0)

    return model


def load_baseline(args):
    from third_party.baselines.networks.resnet_big import MoCoResNet, SupConResNet, SupCEResNet

    invert = False
    class_num = 10
    normalize = True
    image_size = 128

    if args.method == "moco":
        model = MoCoResNet(name=args.model, image_size=image_size, class_num=class_num)
    elif args.method in ["supcon", "simclr"]:
        model = SupConResNet(name=args.model, image_size=image_size)
    else:
        model = SupCEResNet(name=args.model, image_size=image_size)

    ckpt = torch.load(os.path.join(args.path, "ckpt_best.pth"), map_location="cpu")
    state_dict = ckpt["model"]
    old_state_dict = model.state_dict()

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     model.encoder = torch.nn.DataParallel(model.encoder)

        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            if k.startswith("fc") and (k in old_state_dict and v.shape != old_state_dict[k].shape):
                new_state_dict[k] = old_state_dict[k]
            elif k not in old_state_dict:
                pass
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()

        model.load_state_dict(state_dict)

    class ModelWrapper(nn.Module):
        def __init__(self, model, normalize=True, invert=False):
            super(ModelWrapper, self).__init__()
            self.model = model
            if normalize:
                self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            else:
                self.normalize = lambda x: x
            if invert:
                self.invert = lambda x: 1 - x
            else:
                self.invert = lambda x: x

        def forward(self, x):
            x = self.normalize(self.invert(x))
            return self.model(x)

        def get_representation(self, x, rep_type=None):
            x = self.normalize(self.invert(x))
            return self.model.encoder(x)

    update_args(argparser.parse_arguments())
    return ModelWrapper(model, normalize, invert)


def load_geossl(args):
    import third_party.geossl.models as geossl
    from glob import glob

    desc_channels = 256
    min_uncertainty = 1e-8
    normalize = None

    image_size = (128, 128, 3)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    model = geossl.build_model(
        backbone=args.model, image_size=image_size, desc_channels=desc_channels, min_uncertainty=min_uncertainty
    ).to(args.device)

    ckpt_path = sorted(glob(os.path.join(args.path, "checkpoints/*.ckpt")))[-1]
    ckpt = torch.load(ckpt_path, map_location=args.device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class GeoSSLWrapper(nn.Module):
        def __init__(self, model, normalize=None):
            super(GeoSSLWrapper, self).__init__()

            self.model = model
            self.normalize = normalize

        def forward(self, x):
            if self.normalize is not None:
                x = self.normalize(x)
            return self.model(x)

        def get_representation(self, x, rep_type=None):
            x, _ = self(x)

            if rep_type == "flatten":
                x = x.view(x.size(0), -1)
            elif rep_type == "average":
                x = x.mean(dim=(-2, -1))
            else:
                raise NotImplementedError()

            return x
    
    return GeoSSLWrapper(model, normalize)


def load_btcvae(args):
    sys.path.insert(0, "./third_party")
    import third_party.disvae.utils.modelIO as btcvae

    del sys.path[0]

    model = btcvae.load_model(args.path)
    model.eval()

    invert_image = False  # 'mnist' in opt.dataset

    class BetaTCVAEWrapper(nn.Module):
        def __init__(self, model, invert_image=False):
            super(BetaTCVAEWrapper, self).__init__()

            self.model = model
            self.invert_image = invert_image

        def forward(self, x):
            if tuple(x.shape[-3:]) != tuple(self.model.img_size):
                x = F.interpolate(x, size=self.model.img_size[-2:], mode="bilinear", align_corners=True)
            if self.invert_image:
                x = 1 - x

            return model.encoder(x)

        def get_representation(self, x, rep_type=None):
            mu, logvar = self(x)

            if rep_type == "mu":
                x = mu
            elif rep_type == "concat":
                x = torch.cat([mu, logvar], dim=1)
            else:
                raise NotImplementedError()

            return x

    update_args(argparser.parse_arguments())
    return BetaTCVAEWrapper(model, invert_image=invert_image)


def load_ltd(args):
    import sys
    import os

    sys.path.append(os.path.join("third_party", "ltd"))
    from commgame import parse_args, build_model, build_dataloaders

    class SketchModelWrapper(nn.Module):
        def __init__(self, model, normalize=True, invert=False):
            super(SketchModelWrapper, self).__init__()
            self.model = model
            if normalize:
                self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            else:
                self.normalize = lambda x: x.repeat(1, 3, 1, 1)
            if invert:
                self.invert = lambda x: 1 - x
            else:
                self.invert = lambda x: x

        def forward(self, img):
            img = self.normalize(self.invert(img))
            z = self.model.sender_encoder(img)
            p_pos = self.model.decoder.decode_to_params(z)
            pos_order = (p_pos[:, :, 0, 0] < p_pos[:, :, -1, 0]).unsqueeze(-1).unsqueeze(-1)
            p_pos = torch.where(pos_order, p_pos, p_pos.flip(2))

            return p_pos.flatten(1)

        def get_representation(self, x, rep_type=None):
            return self.forward(x)

        def get_reconstruction(self, img):
            img = self.normalize(self.invert(img))
            z = self.model.sender_encoder(img)
            recon = self.model.decoder(z)
            return recon

    invert = False
    normalize = True

    with open(os.path.join(args.path, "train_cmd.txt"), "r") as f:
        train_args = shlex.split(f.read())[1:]
    train_args = parse_args(train_args)
    _ = build_dataloaders(train_args)

    sketch_model = build_model(train_args)
    ckpt = torch.load(os.path.join(args.path, "model_final.pt"), map_location="cpu")["model"]
    sketch_model.load_state_dict(ckpt)

    model = SketchModelWrapper(sketch_model, normalize, invert)

    update_args(argparser.parse_arguments())
    return model


def load_painter(args):
    import sys
    import os

    sys.path.append(os.path.join("third_party", "paint", "inference"))
    import network
    from inference import generate_strokes

    patch_size = 32
    stroke_num = 8
    model_path = os.path.join("third_party", "paint", "inference", "model.pth")

    net_g = network.Painter(5, stroke_num, 256, 8, 3, 3)
    net_g.load_state_dict(torch.load(model_path))
    net_g.eval()
    for param in net_g.parameters():
        param.requires_grad = False

    class PaintModelWrapper(nn.Module):
        def __init__(self, model, invert=False):
            super(PaintModelWrapper, self).__init__()
            self.model = model
            if invert:
                self.invert = lambda x: 1 - x
            else:
                self.invert = lambda x: x

        def forward(self, img):
            img = self.invert(img)
            z = list()
            for img_ in img:
                z.append(generate_strokes(net_g, img_.unsqueeze(0)).flatten())
            return torch.stack(z, dim=0)

        def get_representation(self, x, rep_type=None):
            return self.forward(x)

    invert = False

    update_args(argparser.parse_arguments())
    return PaintModelWrapper(net_g, invert)


def load_clip(args):
    import clip

    model, _ = clip.load("RN101", device=args.device, jit=False)

    class CLIPModelWrapper(nn.Module):
        def __init__(self, model):
            super(CLIPModelWrapper, self).__init__()
            self.model = model
            self.transform_clip = Compose(
                [
                    Resize(224, interpolation=BICUBIC),
                    CenterCrop(224),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )

        def forward(self, img):
            dtype = img.dtype
            img = self.transform_clip(img)
            return self.model.encode_image(img).type(dtype)

        def get_representation(self, x, rep_type=None):
            return self.forward(x)

    update_args(argparser.parse_arguments())
    return CLIPModelWrapper(model)
