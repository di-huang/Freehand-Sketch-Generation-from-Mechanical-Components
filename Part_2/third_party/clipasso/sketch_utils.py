import os
import math

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydiffvg
import skimage
import skimage.io
import torch
import wandb
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import _LRScheduler

from third_party.U2Net_.model import U2NET


def imwrite(img, filename, gamma=2.2, normalize=False, use_wandb=False, wandb_name="", step=0, input_im=None):
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        # repeat along the third dimension
        img = np.expand_dims(img, 2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    img = (img * 255).astype(np.uint8)

    skimage.io.imsave(filename, img, check_contrast=False)
    images = [wandb.Image(Image.fromarray(img), caption="output")]
    if input_im is not None and step == 0:
        images.append(wandb.Image(input_im, caption="input"))
    if use_wandb:
        wandb.log({wandb_name + "_": images}, step=step)


def plot_batch(inputs, outputs, output_dir, step, use_wandb, title):
    plt.figure()
    plt.subplot(2, 1, 1)
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=1)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("inputs")

    plt.subplot(2, 1, 2)
    grid = make_grid(outputs, normalize=False, pad_value=1)
    npgrid = grid.detach().cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("outputs")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"output": wandb.Image(plt)}, step=step)
    plt.savefig("{}/{}".format(output_dir, title))
    plt.close()


def log_input(use_wandb, epoch, inputs, output_dir):
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"input": wandb.Image(plt)}, step=epoch)
    plt.close()
    input_ = inputs[0].cpu().clone().detach().permute(1, 2, 0).numpy()
    input_ = (input_ - input_.min()) / (input_.max() - input_.min())
    input_ = (input_ * 255).astype(np.uint8)
    imageio.imwrite("{}/{}.png".format(output_dir, "input"), input_)


def log_sketch_summary_final(path_svg, use_wandb, device, epoch, loss, title):
    canvas_width, canvas_height, shapes, shape_groups = load_svg(path_svg)
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)

    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    plt.imshow(img.cpu().numpy())
    plt.axis("off")
    plt.title(f"{title} best res [{epoch}] [{loss}.]")
    if use_wandb:
        wandb.log({title: wandb.Image(plt)})
    plt.close()


def log_sketch_summary(sketch, title, use_wandb):
    plt.figure()
    grid = make_grid(sketch.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    if use_wandb:
        wandb.run.summary["best_loss_im"] = wandb.Image(plt)
    plt.close()


def load_svg(path_svg):
    svg = os.path.join(path_svg)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        svg)
    return canvas_width, canvas_height, shapes, shape_groups


def read_svg(path_svg, device, multiply=False):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        path_svg)
    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    return img


def plot_attn_dino(attn, threshold_map, inputs, inds, use_wandb, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(2, attn.shape[0] + 2, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, 2)
    plt.imshow(attn.sum(0).numpy(), interpolation='nearest')
    plt.title("atn map sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 3)
    plt.imshow(threshold_map[-1].numpy(), interpolation='nearest')
    plt.title("prob sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 4)
    plt.imshow(threshold_map[:-1].sum(0).numpy(), interpolation='nearest')
    plt.title("thresh sum")
    plt.axis("off")

    for i in range(attn.shape[0]):
        plt.subplot(2, attn.shape[0] + 2, i + 3)
        plt.imshow(attn[i].numpy())
        plt.axis("off")
        plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 1 + i + 4)
        plt.imshow(threshold_map[i].numpy())
        plt.axis("off")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()


def plot_attn_clip(attn, threshold_map, inputs, inds, use_wandb, output_path, display_logs):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("atn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
        (threshold_map.max() - threshold_map.min())
    plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()


def plot_atten(attn, threshold_map, inputs, inds, output_path, use_wandb=False, saliency_model="clip", display_logs=False):
    if saliency_model == "dino":
        plot_attn_dino(attn, threshold_map, inputs,
                       inds, use_wandb, output_path)
    elif saliency_model == "clip":
        plot_attn_clip(attn, threshold_map, inputs, inds,
                       use_wandb, output_path, display_logs)


def fix_image_scale(im):
    im_np = np.array(im) / 255
    height, width = im_np.shape[0], im_np.shape[1]
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    new_background[y: y + height, x: x + width] = im_np
    new_background = (new_background / new_background.max()
                      * 255).astype(np.uint8)
    new_im = Image.fromarray(new_background)
    return new_im


net = None

def get_mask_u2net(args, pil_im, save_png=True):
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(
            0.26862954, 0.26130258, 0.27577711)),
    ])

    input_im_trans = data_transforms(pil_im).unsqueeze(0).to(args.device)
    model_dir = os.path.join("./third_party/U2Net_/saved_models/u2net.pth")
    net = U2NET(3, 1)
    if torch.cuda.is_available() and args.use_gpu:
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.detach())
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    # predict = 1 - predict
    mask = torch.cat([predict, predict, predict], axis=0).permute(1, 2, 0)
    mask = mask.cpu().numpy()
    # predict_np = predict.clone().cpu().data.numpy()
    if save_png:
        im = Image.fromarray((mask[:, :, 0]*255).astype(np.uint8)).convert('RGB')
        im.save(f"{args.output_dir}/mask.png")

    im_np = np.array(pil_im) / 255
    mask_alpha = 0.0
    im_np = mask * im_np + (1 - mask) * ((1 - mask_alpha) + mask_alpha * im_np)
    im_final = (im_np * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)

    return im_final, predict

def get_u2net(args):
    model_dir = os.path.join("./third_party/U2Net_/saved_models/u2net.pth")
    net = U2NET(3, 1)
    if torch.cuda.is_available() and args.use_gpu:
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    return net

def get_mask_u2net_batch(args, img, net=None, return_foreground=True, return_background=False):
    if net is None:
        net = get_u2net(args)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(img.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(img.device)

    input_im_trans = (img - mean) / std
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.detach())
        
    pred = d1[:, 0, :, :]
    pred_min = pred.view(pred.size(0), -1).min(dim=-1)[0].view(-1, 1, 1)
    pred_max = pred.view(pred.size(0), -1).max(dim=-1)[0].view(-1, 1, 1)
    pred = (pred - pred_min) / (pred_max - pred_min)
    predict = pred
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    mask = predict.unsqueeze(1)
    mask_inv = 1 - mask

    foreground = background = None
    if return_foreground:
        foreground = mask * img + mask_inv
    if return_background:
        background = mask_inv * img + mask

    return foreground, background, predict, net

def mask_image(args, im):
    global net

    if net is None:
        model_dir = os.path.join("./third_party/U2Net_/saved_models/u2net.pth")
        net = U2NET(3, 1)
        if torch.cuda.is_available() and args.use_gpu:
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        net.eval()

    data_transforms = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(data_transforms(im).detach())
    pred = d1[:, 0, :, :]

    pred_min = pred.flatten(1).min(dim=1)[0].view(-1, 1, 1)
    pred_max = pred.flatten(1).max(dim=1)[0].view(-1, 1, 1)
    predict = (pred - pred_min) / (pred_max - pred_min + 1e-6)

    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    # predict = 1 - predict
    mask = torch.stack([predict, predict, predict], axis=1)
    mask_alpha = 0.0
    im_final = mask * im + (1 - mask) * ((1 - mask_alpha) + mask_alpha * im)

    return im_final
    

def check_nan(tensor):
    if (tensor != tensor).sum() != 0:
        return True
    return False

def set_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
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
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

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
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cycle(dataloader, distributed=False):
    epoch = 0
    while True:
        for data in dataloader:
            yield data
        epoch += 1
        if distributed:
            dataloader.sampler.set_epoch(epoch)

class ImageGrid:
    def __init__(self, num_img=10, nrow=6):
        self.num_img = num_img * nrow
        self.nrow = nrow
        self._figures = []

    def update(self, *images):
        num_grid = len(images)
        fig, axs = plt.subplots(1, num_grid, figsize=(4 * num_grid, 6), constrained_layout=True,
                                gridspec_kw={"wspace": 0.01})
        for idx, img in enumerate(images):
            grid = make_grid(img[:self.num_img].detach().cpu(), nrow=self.nrow).permute(1, 2, 0).numpy()
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