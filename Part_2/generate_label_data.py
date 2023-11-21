import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from third_party.clipasso.models.painter_params import Painter, PainterOptimizer
from third_party.clipasso.models.loss import Loss
from third_party.clipasso import sketch_utils as utils
from argparser import parse_arguments
import argparse

import numpy as np
from PIL import Image

import os
import pickle
import time
import random
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import subprocess as sp
import svgwrite
from xml.dom import minidom

parser = argparse.ArgumentParser()

# general arguments
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--dataset', type=str, help='clevr_train, clevr_val, stl10_train+unlabeled, stl10_test, etc..')
parser.add_argument('--data_root', type=str, help='the path to the root directory containing datasets to process.')
parser.add_argument('--img_paths', type=str, nargs='+', help='image file-paths (with wildcards) to process.')
parser.add_argument('--key_steps', type=int, nargs='+',
                    default=[0, 50, 100, 200, 400, 700, 1000, 1500, 2000])
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--num_generation', type=int, default=-1, help='number of sketches to generate, generate all sketches in the dataset if -1')
parser.add_argument('--chunk', type=int, nargs=2, help='--chunk (num_chunks) (chunk_index)')
parser.add_argument('--init_mode', type=str, default='lbs', help='lbs: LBS first; sam: SAM first')
parser.add_argument('--sam_init_num', type=int, default=3, help='the number of initial points for each contour if using SAM to initialize stroke points')
parser.add_argument('--option_mask_image', type=int, default=0, help='get mask of image or not')

# optimization arguments
parser.add_argument('--width', type=float, default=1.5, help='foreground-stroke width')
parser.add_argument('--width_bg', type=float, default=8.0, help='background-stroke width')
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1.0)

# extra arguments
parser.add_argument('--no_tqdm', action='store_true')
parser.add_argument('--no_track_time', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--save_svgs', action='store_true')

# CLIPasso arguments
parser.add_argument("--target", type=str)
parser.add_argument("--path_svg", type=str, default="none")
parser.add_argument("--use_wandb", type=int, default=0)
parser.add_argument("--num_iter", type=int, default=500)
parser.add_argument("--num_stages", type=int, default=1)
parser.add_argument("--color_vars_threshold", type=float, default=0.0)
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--control_points_per_seg", type=int, default=4)
parser.add_argument("--attention_init", type=int, default=1)
parser.add_argument("--saliency_model", type=str, default="clip")
parser.add_argument("--saliency_clip_model", type=str, default="ViT-B/32")
parser.add_argument("--xdog_intersec", type=int, default=0)
parser.add_argument("--mask_object_attention", type=int, default=0)
parser.add_argument("--softmax_temp", type=float, default=0.3)
parser.add_argument("--percep_loss", type=str, default="none")
parser.add_argument("--train_with_clip", type=int, default=0)
parser.add_argument("--clip_weight", type=float, default=0)
parser.add_argument("--start_clip", type=int, default=0)
parser.add_argument("--num_aug_clip", type=int, default=3) # LBS: 4
parser.add_argument("--include_target_in_aug", type=int, default=0) # unused
parser.add_argument("--augment_both", type=int, default=1) # unused
parser.add_argument("--augemntations", type=str, default="affine")
parser.add_argument("--noise_thresh", type=float, default=0.5)
parser.add_argument("--force_sparse", type=float, default=1)
parser.add_argument("--clip_conv_loss", type=float, default=1)
parser.add_argument("--clip_conv_loss_type", type=str, default="L2")
parser.add_argument("--clip_model_name", type=str, default="RN101")
parser.add_argument("--clip_fc_loss_weight", type=float, default=0.1)
parser.add_argument("--clip_text_guide", type=float, default=0)
parser.add_argument("--text_target", type=str, default="none")


abs_path = os.path.abspath(os.getcwd())
if not os.path.isfile(f"{abs_path}/third_party/U2Net_/saved_models/u2net.pth"):
    sp.run(["gdown", "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
            "-O", "third_party/U2Net_/saved_models/"])

# Builds a dataset with the specified image file-paths.
class ImageDataset(Dataset):
    def __init__(self, path_formats, transform=None):
        self.path_formats = path_formats
        self.transform = transform

        self.image_paths = []
        for path_format in self.path_formats:
            self.image_paths += glob(path_format, recursive='**' in path_format)
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # dummy label
        return [image, image_path], -1


# Prefixes the index and drops the label for each sample.
class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index][0]


# The main class that generates sketch data with the specified image dataset.
class DataGenerator(nn.Module):
    def __init__(self, args):
        super(DataGenerator, self).__init__()
        
        self.args = args
        
        # Initialize the renderers.
        renderers = []
        renderers.append(Painter(
            args,
            args.num_strokes, args.num_segments,
            imsize=args.image_scale,
            device=args.device,
        ))
        for _ in range(args.batch_size - 1):
            renderers.append(Painter(
                args,
                args.num_strokes, args.num_segments,
                imsize=args.image_scale,
                device=args.device,
                clip_model=renderers[0].clip_model,
                clip_preprocess=renderers[0].clip_preprocess,
                dino_model=renderers[0].dino_model
            ))

        self.renderers = nn.ModuleList(renderers)
        self.criterion = Loss(args)
        self.u2net = utils.get_u2net(args)

    def save_sample_visualization(self, sample_name, sample_image, sample_paths):
        fig, axs = plt.subplots(1, len(sample_paths)+1, figsize=(3+3*len(sample_paths), 3))

        axs[0].set_title('image')
        axs[0].imshow(sample_image)

        t = np.linspace(0, 1, 10)
        for i, (step, step_paths) in enumerate(sample_paths.items()):
            curves = cubic_bezier(step_paths['pos'], t)
            axs[i+1].set_title(f'step {step}')
            for curve, color, width in zip(curves, step_paths['color'], step_paths['radius']):
                axs[i+1].plot(*curve.T[::-1], c=color)
            axs[i+1].set_ylim(1,-1)
            axs[i+1].tick_params(axis='both', which='major', labelsize=5)

        vis_filename = os.path.join(self.args.output_dir, f'vis/{sample_name}.jpg')
        fig.savefig(vis_filename)
        plt.close()

    def save_svgs(self, sample_paths, image_path):
        image_path_list = os.path.normpath(image_path).split(os.path.sep)[1:]
        specific_path = os.path.splitext(os.path.sep.join(image_path_list))[0]
        image_path = os.path.join(self.args.output_dir, 'svgs', specific_path)
        os.makedirs(image_path, exist_ok=True)

        for i, (step, step_paths) in enumerate(sample_paths.items()):
            strokes_ = []
            for stroke in step_paths['pos']:
                x1, y1, x2, y2, x3, y3, x4, y4 = stroke
                # stroke_ = [(x1 + 2, y1 + 2), (x2 + 2, y2 + 2), (x3 + 2, y3 + 2), (x4 + 2, y4 + 2)]
                # stroke_ = [(y1 + 2, x1 + 2), (y2 + 2, x2 + 2), (y3 + 2, x3 + 2), (y4 + 2, x4 + 2)]
                stroke_ = [y1 + 2, x1 + 2, y2 + 2, x2 + 2, y3 + 2, x3 + 2, y4 + 2, x4 + 2]
                strokes_.append(stroke_)

            # -------------------------------------------------

            svg_out_file_path = os.path.join(image_path, f'{step}.svg')
            dwg = svgwrite.Drawing(svg_out_file_path, profile='full', size=(200, 200), viewBox="0 0 4 4", version="1.2")
            g = dwg.add(dwg.g())
            for stroke in strokes_:
                d = f'M {stroke[0]} {stroke[1]} C {stroke[2]} {stroke[3]} {stroke[4]} {stroke[5]} {stroke[6]} {stroke[7]}'
                g.add(dwg.path(d=d, stroke_width="0.05", fill="none", stroke="rgb(0, 0, 0)", stroke_opacity="1.0", stroke_linecap="round", stroke_linejoin="round"))
            dom = minidom.parseString(dwg.tostring())
            with open(svg_out_file_path, 'w', encoding='utf-8') as f:
                f.write(dom.toprettyxml(indent="  "))

            # -------------------------------------------------

            # svg_header = '''<?xml version="1.0" ?>
            # <svg xmlns="http://www.w3.org/2000/svg" version="1.2" baseProfile="full" viewBox="0 0 4 4" width="200" height="200">
            # <defs />
            # <g>
            # '''

            # svg_footer = '''  </g>
            # </svg>'''

            # paths = []
            # for stroke in strokes_:
            #     path_data = f'M {stroke[0]} {stroke[1]} C {stroke[2]} {stroke[3]} {stroke[4]} {stroke[5]} {stroke[6]} {stroke[7]}'
            #     path_str = f'    <path d="{path_data}" stroke-width="0.05" fill="none" stroke="rgb(0, 0, 0)" stroke-opacity="1.0" stroke-linecap="round" stroke-linejoin="round"/>'
            #     paths.append(path_str)

            # final_svg = svg_header + '\n'.join(paths) + svg_footer
            # dom = minidom.parseString(final_svg)

            # svg_out_file_path = os.path.join(image_path, f'{step}.svg')
            # with open(svg_out_file_path, 'w') as file:
            #     file.write(dom.toprettyxml(indent="   "))

    def _generate(self, args, image, image_path, mask, num_iter, num_strokes, width, attn_colors, path_dicts=None, gradual_colors=True, use_tqdm=False):
        curr_batch_size = image.size(0)
        if path_dicts is None:
            path_dicts = [None] * curr_batch_size
        renderers = self.renderers[:curr_batch_size]

        for renderer, curr_image, curr_image_path, curr_mask, path_dict in zip(renderers, image, image_path, mask, path_dicts):
            renderer.set_random_noise(0)
            try:
                renderer.init_image(
                    target_im=curr_image.unsqueeze(0),
                    mask=curr_mask.unsqueeze(0) if curr_mask else None, # unused
                    stage=0,
                    randomize_colors=False,
                    attn_colors=attn_colors,
                    attn_colors_stroke_sigma=5.0,
                    path_dict=path_dict,
                    new_num_strokes=num_strokes,
                    new_width=width,
                    init_settings=(args.init_mode, args.sam_init_num)
                )
            except Exception as err:
                print(err, curr_image_path)
                assert(False)

        if num_iter == 0:
            for renderer in renderers:
                for key_step in self.args.key_steps:
                    renderer.log_shapes(str(key_step))
            path_dicts = [renderer.path_dict_np(radius=width) for renderer in renderers]
            if gradual_colors:
                for sample_paths in path_dicts:
                    ts = np.linspace(0, 1, len(sample_paths))
                    for t, step_paths in zip(ts, sample_paths.values()):
                        step_paths['color'] *= t
            return path_dicts

        optimizer = PainterOptimizer(self.args, renderers)
        optimizer.init_optimizers()

        steps = range(num_iter)
        if use_tqdm:
            steps = tqdm(steps)

        for step in steps:
            for renderer in renderers:
                renderer.set_random_noise(step)

            optimizer.zero_grad_()
            sketches = torch.cat([renderer.get_image().to(self.args.device) for renderer in renderers], dim=0)
            loss = sum(self.criterion(sketches, image.detach(), step, points_optim=optimizer).values()).mean()
            loss.backward()
            optimizer.step_(optimize_points=True, optimize_colors=False)

            if (step+1) in self.args.key_steps:
                for renderer in renderers:
                    renderer.log_shapes()
                    renderer.log_shapes(str(step+1))

        return [renderer.path_dict_np(radius=width) for renderer in renderers]

    def generate_for_batch(self, args, index, image_info, use_tqdm=False):
        image, image_path = image_info
        sample_names = [f'{idx}_{self.args.seed}' for idx in index.tolist()]

        # NOTE: mainly generate mask stuffs using u2net here, and our 'foreground' is set to be 'image' by default
        if args.option_mask_image:
            foreground, background, mask, _ = utils.get_mask_u2net_batch(self.args, image, net=self.u2net, return_background=True)
            with torch.no_grad():
                mask_areas = mask.view(mask.size(0), -1).mean(dim=1).tolist()
                mask_areas = dict(zip(sample_names, mask_areas))
        else:
            foreground = image # not mask image by default
            mask = len(foreground) * [None]
            mask_areas = {}

        num_strokes_fg = self.args.num_strokes - self.args.num_background
        num_strokes_bg = self.args.num_background
        stroke_width_fg = self.args.width
        stroke_width_bg = self.args.width_bg

        # NOTE: generate label data in vector graphics format here, mask is unused
        path_dicts = self._generate(args, foreground, image_path, mask, self.args.num_iter, num_strokes_fg, stroke_width_fg, False, use_tqdm=use_tqdm)
        new_path_dicts = []
        for item1, item2 in zip(path_dicts, image_path):
            new_item = {'iterations': item1, 'img_path': item2}
            new_path_dicts.append(new_item)
        path_dicts = new_path_dicts

        # NOTE: we return here, mask_areas is unused
        if not self.args.enable_color:
            path_dicts = dict(zip(sample_names, path_dicts))
            return path_dicts, mask_areas

        color_dicts = self._generate(args, foreground, image_path, mask, 0, None, stroke_width_fg, True, path_dicts=path_dicts, use_tqdm=use_tqdm)
        for paths, colors in zip(path_dicts, color_dicts):
            for step in self.args.key_steps:
                step = str(step)
                paths[step]['color'] = colors[step]['color']

        if num_strokes_bg <= 0:
            path_dicts = dict(zip(sample_names, path_dicts))
            return path_dicts, mask_areas
        
        path_dicts_bg = self._generate(args, background, image_path, 1 - mask, 0, num_strokes_bg, stroke_width_bg, True, use_tqdm=use_tqdm)
        for paths, paths_bg in zip(path_dicts, path_dicts_bg):
            for step in self.args.key_steps:
                step = str(step)
                paths[step]['pos'] = np.concatenate([paths[step]['pos'], paths_bg[step]['pos']], axis=0)
                paths[step]['color'] = np.concatenate([paths[step]['color'], paths_bg[step]['color']], axis=0)
                if 'radius' in paths[step]:
                    paths[step]['radius'] = np.concatenate([paths[step]['radius'], paths_bg[step]['radius']], axis=0)

        path_dicts = dict(zip(sample_names, path_dicts))
        return path_dicts, mask_areas
    
    def generate_for_dataset(self, args, dataloader, use_tqdm=False, track_time=False):
        path_dicts = {}
        mask_areas = {}

        if track_time:
            start_time = time.time()

        min_index = next(iter(dataloader.dataset.indices))
        max_index = next(iter(reversed(dataloader.dataset.indices)))

        generated_samples = 0
        total_samples = len(dataloader.dataset)
        
        for index, image_info in dataloader:
            image, image_path = image_info
            if track_time:
                print(f'generating samples for {index.min().item()}..{index.max().item()} of {min_index}..{max_index}:')
            image = image.to(self.args.device)
            
            batch_path_dicts, batch_mask_areas = self.generate_for_batch(args, index, (image, image_path), use_tqdm=use_tqdm)
            path_dicts.update(batch_path_dicts)
            mask_areas.update(batch_mask_areas)

            if self.args.visualize:
                for i, (sample_name, sample_paths) in enumerate(batch_path_dicts.items()):
                    sample_image = image[i].detach().cpu().permute(1, 2, 0).numpy()
                    sample_paths_to_show = sample_paths['iterations']
                    vis_file_name = sample_name + '==' + os.path.splitext(os.path.basename(sample_paths['img_path']))[0]
                    self.save_sample_visualization(vis_file_name, sample_image, sample_paths_to_show)

            if self.args.save_svgs:
                for i, (sample_name, sample_paths) in enumerate(batch_path_dicts.items()):
                    self.save_svgs(sample_paths['iterations'], sample_paths['img_path'])

            if track_time:
                generated_samples += image.size(0)
                completion = generated_samples / total_samples
                time_passed = time.time() - start_time
                time_left = time_passed / completion - time_passed
                tp_days, tp_hours, tp_minutes, _ = to_dhms(time_passed)
                tl_days, tl_hours, tl_minutes, _ = to_dhms(time_left)
                print(
                    f'{completion*100:.02f}% ({generated_samples}/{total_samples}) complete.'\
                    f' {tp_days}d {tp_hours}h {tp_minutes}m passed.'\
                    f' expected {tl_days}d {tl_hours}h {tl_minutes}m left.'
                )

        if track_time:
            print(f'took {time_passed:.02f}s to generate {total_samples} samples.')

        return path_dicts, mask_areas


def cubic_bezier(p, t):
    p = p.reshape(-1, 4, 1, 2)
    t = t.reshape(1, -1, 1)
    return ((1-t)**3)*p[:,0] + 3*((1-t)**2)*t*p[:,1] + 3*(1-t)*(t**2)*p[:,2] + (t**3)*p[:,3]

def to_dhms(seconds):
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds


def get_dataset(args):
    assert (args.img_paths is not None) ^ (args.dataset is not None and args.data_root is not None),\
        "either \'img_paths\' or \'dataset\' and \'data_root\' must be specified!"

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # edit here to add custom datasets to process.
    if args.dataset is not None:
        if args.dataset.startswith('stl10_'):
            _, split = args.dataset.split('stl10_')
            dataset = datasets.STL10(args.data_root, transform=transform, download=True, split=split)
        elif args.dataset.startswith('clevr_'):
            _, split = args.dataset.split('clevr_')
            img_path = os.path.join(args.data_root, f'clevr/images/{split}/CLEVR_{split}_*.png')
            dataset = ImageDataset([img_path], transform=transform)
    else:
        dataset = ImageDataset(args.img_paths, transform=transform)

    num_generation = len(dataset) if args.num_generation == -1 else args.num_generation
    if args.chunk is not None:
        num_chunks, chunk_index = args.chunk
        chunk_size = int(np.ceil(num_generation / num_chunks))
        chunk_start = chunk_size * chunk_index
        chunk_end = min(chunk_start + chunk_size, num_generation)
    else:
        chunk_start = 0
        chunk_end = num_generation

    dataset = IndexedDataset(dataset)
    dataset = Subset(dataset, range(chunk_start, chunk_end))

    return dataset


def main(args=None):
    if args is None:
        args = parse_arguments()
        args.update(vars(parser.parse_known_args()[0]))
    args.num_iter = max(args.key_steps)
    args.use_gpu = not args.no_cuda
    args.image_scale = args.image_size
    args.color_lr = 0.01

    dataset = get_dataset(args)
    print('--- batch_size:', args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)

    data_generator = DataGenerator(args).to(args.device)
    path_dicts, mask_areas = data_generator.generate_for_dataset(args, dataloader, use_tqdm=not args.no_tqdm, track_time=not args.no_track_time)

    chunk_info = f'_{args.chunk[1]}' if args.chunk else ''
    with open(os.path.join(args.output_dir, f'path{chunk_info}.pkl'), 'wb') as file:
        pickle.dump(path_dicts, file)

    if args.option_mask_image and mask_areas:
        with open(os.path.join(args.output_dir, f'maskareas_seed{args.seed}{chunk_info}.pkl'), 'wb') as file:
            pickle.dump(mask_areas, file)


if __name__ == '__main__':
    main()
