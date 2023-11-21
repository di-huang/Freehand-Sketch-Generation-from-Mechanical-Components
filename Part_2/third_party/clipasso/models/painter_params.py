import random
from ..CLIP_ import clip
import numpy as np
import pydiffvg
from .. import sketch_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
import copy

from . import sam

class Painter(torch.nn.Module):
    def __init__(self, args,
                num_strokes=4,
                num_segments=4,
                imsize=224,
                device=None, clip_model=None, clip_preprocess=None, dino_model=None, control_radius=-1.0):
        super(Painter, self).__init__()

        self.args = args
        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg
        self.opacity_optim = args.force_sparse
        self.num_stages = args.num_stages
        self.add_random_noise = "noise" in args.augemntations
        self.noise_thresh = args.noise_thresh
        self.softmax_temp = args.softmax_temp

        self.device = device
        self.canvas_width, self.canvas_height = imsize, imsize
        self.color_vars_threshold = args.color_vars_threshold

        self.path_svg = args.path_svg
        self.strokes_per_stage = self.num_paths

        # attention related for strokes initialisation
        self.attention_init = args.attention_init
        self.target_path = args.target
        self.saliency_model = args.saliency_model
        self.xdog_intersec = args.xdog_intersec
        self.mask_object = args.mask_object_attention
        
        self.text_target = args.text_target # for clip gradients
        self.saliency_clip_model = args.saliency_clip_model

        self.control_radius = control_radius
        self.constraint_controls = control_radius > 0.0

        self.strokes_counter = 0 # counts the number of calls to "get_path"        
        self.epoch = 0
        # self.final_epoch = args.num_iter - 1

        if clip_model is None:
            self.clip_model, self.clip_preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
            self.clip_model.eval().to(self.device)
        else:
            assert clip_preprocess is not None
            self.clip_model = clip_model
            self.clip_preprocess = clip_preprocess

        assert self.saliency_model in ["dino", "clip"]
        
        self.dino_model = None
        if self.saliency_model == "dino":
            if dino_model is None:
                self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').eval().to(self.device)
            else:
                self.dino_model = dino_model
        

    def init_image(self, target_im=None, mask=None, stage=0, randomize_colors=False, attn_colors=False, attn_colors_stroke_sigma=-1.0, path_dict=None, return_rgba=False, new_num_strokes=None, new_width=None, init_settings=('lbs', 0)):
        if new_num_strokes is not None:
            self.num_paths = new_num_strokes
            self.strokes_per_stage = self.num_paths
        if new_width is not None:
            self.width = new_width

        self.strokes_counter = 0 # counts the number of calls to "get_path"        
        self.epoch = 0
        self.shapes = []
        self.shape_groups = []
        self.points_vars = []
        self.color_vars = []
        self.optimize_flag = []

        if self.constraint_controls:
            self.pivots = []
            self.controls = []
        
        self.define_attention_input(target_im)
        self.mask = mask
        self.attention_map = self.set_attention_map() if self.attention_init else None
        self.thresh = self.set_attention_threshold_map() if self.attention_init else None

        # start ------------------------------------------

        init_mode, sam_init_num = init_settings
        inds_init_size = len(self.inds)
        pts_to_be_added_grouped, pts_to_be_added_sorted = sam.add_init_points_by_sam(target_im, self.device, sam_init_num, self.thresh, self.inds)
        pts_to_be_added_sorted = np.array(pts_to_be_added_sorted)

        # ------------------------------------------
        
        outlier_indices = sam.get_outlier_indices(target_im, pts_to_be_added_sorted)
        pts_to_be_added_sorted = np.delete(pts_to_be_added_sorted, outlier_indices, axis=0)
        sam_pts_size = len(pts_to_be_added_sorted)

        outlier_indices = sam.get_outlier_indices(target_im, self.inds)
        inds_outlier_size = len(outlier_indices)

        # ------------------------------------------

        if init_mode == 'sam':
            # Warning: there may be outlier strokes if using SAM-first mode
            if sam_pts_size >= inds_init_size:
                self.inds = np.array(pts_to_be_added_sorted[:inds_init_size])
            else:
                temp_inds = np.array(self.inds)
                if inds_outlier_size > 0:
                    outlier_to_keep = temp_inds[outlier_indices]
                    temp_inds = np.delete(temp_inds, outlier_indices, axis=0)
                    temp_inds = np.concatenate((temp_inds, outlier_to_keep))
                self.inds = np.array(pts_to_be_added_sorted)
                temp_size_ = inds_init_size - sam_pts_size
                self.inds = np.concatenate((self.inds, temp_inds[:temp_size_]))
        else:
            if sam_pts_size >= inds_outlier_size:
                if inds_outlier_size > 0:
                    self.inds = np.delete(self.inds, outlier_indices, axis=0)
                    self.inds = np.concatenate((self.inds, pts_to_be_added_sorted[:inds_outlier_size]))
            else:
                self.inds = np.delete(self.inds, outlier_indices, axis=0)
                self.inds = np.concatenate((self.inds, pts_to_be_added_sorted[:sam_pts_size]))
                size_to_add = inds_init_size - len(self.inds)
                extra_pts_to_add = []
                for k in range(size_to_add):
                    extra_pts_to_add.append(sam.random_point_around(self.inds[k % len(self.inds)]))
                extra_pts_to_add = np.array(extra_pts_to_add)
                self.inds = np.concatenate((self.inds, extra_pts_to_add))

        # ------------------------------------------

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()

        # generate logs for stroke init
        utils.plot_atten(self.attention_map, self.thresh, target_im, np.array(self.inds), "logs/my_data/init.jpg")

        assert(len(self.inds) == len(self.inds_normalised) == inds_init_size)

        # end ------------------------------------------

        assert(not randomize_colors or not attn_colors)

        def get_color(path):
            return self.get_color(path=path if attn_colors else None, path_blur_sigma=attn_colors_stroke_sigma, random=randomize_colors)

        if path_dict is None:
            if stage > 0:
                # if multi stages training than add new strokes on existing ones
                # don't optimize on previous strokes
                self.optimize_flag = [False for i in range(len(self.shapes))]

                for i in range(self.strokes_per_stage):
                    path = self.get_path()
                    stroke_color = get_color(path)
                    self.shapes.append(path)
                    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                        fill_color = None,
                                                        stroke_color = stroke_color)
                    self.shape_groups.append(path_group)
                    self.optimize_flag.append(True)

            else:
                num_paths_exists = 0
                if self.path_svg != "none":
                    self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = utils.load_svg(self.path_svg)
                    # if you want to add more strokes to existing ones and optimize on all of them
                    num_paths_exists = len(self.shapes)

                for i in range(num_paths_exists, len(self.inds_normalised)):
                    path = self.get_path()
                    stroke_color = get_color(path)
                    self.shapes.append(path)
                    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                        fill_color = None,
                                                        stroke_color = stroke_color)
                    self.shape_groups.append(path_group)        
                self.optimize_flag = [True for i in range(len(self.shapes))]
        else:
            path_dict = self._deserialize_path_dict(path_dict)

            for log_idx, paths in path_dict.items():
                points_list = paths['pos']
                stroke_colors = paths['color']
                    
                for i_points, (points, stroke_color) in enumerate(zip(points_list, stroke_colors)):
                    path = self.get_path(points=points)

                    if randomize_colors or attn_colors:
                        stroke_color = get_color(path)

                    self.shapes.append(path)
                    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(self.shapes) - 1]),
                                                        fill_color = None,
                                                        stroke_color = stroke_color)
                    self.shape_groups.append(path_group)        
                self.optimize_flag = [True for i in range(len(self.shapes))]

        if self.constraint_controls:
            for path in self.shapes:
                self.pivots.append(path.points[[0, 3]].detach())
                self.controls.append(path.points[[1, 2]].detach())
                # self.controls.append((path.points[[1, 2]] - path.points[[0, 3]]).detach() / self.control_radius)
                # self.controls.append(torch.zeros_like(path.points[[0, 3]]))

        self.shapes_log = {
            '0': copy.deepcopy(self.shapes),
            'best': copy.deepcopy(self.shapes)
        }
        self.color_log = {
            '0': copy.deepcopy(self.shape_groups),
            'best': copy.deepcopy(self.shape_groups)
        }
        
        img = self.render_warp()
        if not return_rgba:
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - img[:, :, 3:4])
            img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        return img
        # utils.imwrite(img.cpu(), '{}/init.png'.format(args.output_dir), gamma=args.gamma, use_wandb=args.use_wandb, wandb_name="init")

    def log_shapes(self, idx='best'):
        if idx == 'best':
            self.shapes_log[idx] = copy.deepcopy(self.shapes)
            self.color_log[idx] = copy.deepcopy(self.shape_groups)
        else:
            self.shapes_log[idx] = copy.deepcopy(self.shapes_log['best'])
            self.color_log[idx] = copy.deepcopy(self.color_log['best'])
    
    def get_color(self, path=None, path_blur_sigma=-1.0, random=False):
        color = torch.zeros((4, ))
        color[3] = 1.0

        if random:
            color[:3].uniform_()
        elif path is not None:
            def rasterize_single_stroke(path, sigma):
                with torch.no_grad():
                    path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                                        fill_color = None,
                                                        stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0]))
                    scene_args = pydiffvg.RenderFunction.serialize_scene(\
                        self.canvas_width, self.canvas_height, [path], [path_group])
                    img = pydiffvg.RenderFunction.apply(self.canvas_width, # width
                                self.canvas_height, # height
                                2,   # num_samples_x
                                2,   # num_samples_y
                                0,   # seed
                                None,
                                *scene_args)
                    img = img[..., 3].cpu().numpy()
                    if path_blur_sigma > 0:
                        img = gaussian_filter(img, path_blur_sigma)
                    return img
            
            def avoid_zero(x, eps=1e-8):
                return np.clip(x, eps, x.max() + eps)

            stroke_area = rasterize_single_stroke(path, path_blur_sigma)
            attn_area = self.inds_attn_map
            # mask_area = (self.inds_image_input < 1.0).all(axis=-1).astype(np.float32)
            receptive_field = avoid_zero(stroke_area) * avoid_zero(attn_area) # * avoid_zero(mask_area)
            receptive_field /= receptive_field.sum()
            color[:3] = torch.from_numpy((receptive_field[:, :, None] * self.inds_image_input).reshape(-1, 3).sum(axis=0))

        return color

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device) # NHWC -> NCHW
        return img

    def get_path(self, points=None):
        self.num_control_points = torch.zeros(self.num_segments, dtype = torch.int32) + (self.control_points_per_seg - 2)

        if points is None:
            points = []
            p0 = self.inds_normalised[self.strokes_counter] if self.attention_init else (random.random(), random.random())
            points.append(p0)

            for j in range(self.num_segments):
                radius = 0.05
                for k in range(self.control_points_per_seg - 1):
                    p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                    points.append(p1)
                    p0 = p1
            points = torch.tensor(points).to(self.device)
            points[:, 0] *= self.canvas_width
            points[:, 1] *= self.canvas_height
        else:
            points = points.to(self.device)
        
        path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width),
                                is_closed = False)
        self.strokes_counter += 1
        return path

    def render_warp(self):
        if self.opacity_optim:
            for group in self.shape_groups:
                group.stroke_color.data[:3].clamp_(0., 1.) # to force black stroke
                group.stroke_color.data[-1].clamp_(1., 1.) # opacity
                # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
        _render = pydiffvg.RenderFunction.apply
        # uncomment if you want to add random noise
        if self.add_random_noise:
            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.canvas_width, self.canvas_height)
                for path in self.shapes:
                    path.points.data.add_(eps * torch.randn_like(path.points))
        if self.constraint_controls:
            shapes = []
            self.shapes = []
            
            for pivots, controls in zip(self.pivots, self.controls):
                points = torch.cat([
                    pivots[:1],
                    pivots + self.control_radius * torch.tanh((controls - pivots) / self.control_radius),
                    pivots[1:]
                ], dim=0)

                path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points,
                                stroke_width = torch.tensor(self.width),
                                is_closed = False)
                shapes.append(path)

                path = pydiffvg.Path(num_control_points = self.num_control_points,
                                points = points.detach(),
                                stroke_width = torch.tensor(self.width),
                                is_closed = False)
                self.shapes.append(path)

            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height, shapes, self.shape_groups)
        else:
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = _render(self.canvas_width, # width
                    self.canvas_height, # height
                    2,   # num_samples_x
                    2,   # num_samples_y
                    0,   # seed
                    None,
                    *scene_args)
        return img.to(self.device)
    
    def parameters(self):
        self.points_vars = []
        # storkes' location optimization
        if self.constraint_controls:
            for i, (path, pivots, controls) in enumerate(zip(self.shapes, self.pivots, self.controls)):
                if self.optimize_flag[i]:
                    # path.points.requires_grad = True
                    # self.points_vars.append(path.points)
                    pivots.requires_grad_(True)
                    controls.requires_grad_(True)
                    self.points_vars.append(pivots)
                    self.points_vars.append(controls)
        else:
            for i, path in enumerate(self.shapes):
                if self.optimize_flag[i]:
                    path.points.requires_grad = True
                    self.points_vars.append(path.points)
        return self.points_vars
    
    def get_points_parans(self):
        return self.points_vars

    def set_points_to_best(self):
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.data = self.best_shapes[i].points.data        
    
    def set_color_parameters(self):
        # for storkes' color optimization (opacity)
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)
        return self.color_vars

    def get_color_parameters(self):
        return self.color_vars
        
    def save_svg(self, output_dir, name):
        pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)

    def _deserialize_path_dict(self, path_dict, last_index_only=True):
        des_path_dict = {}

        if last_index_only:
            indices = list(path_dict.keys())
            last_index = max(indices, key=int)
            path_dict = {last_index: path_dict[last_index]}

        for log_idx, path in path_dict.items():
            if 'pos' not in path or 'color' not in path:
                continue
            
            des_path_dict[log_idx] = {'pos': [], 'color': []}
            
            for idx, (pos, col) in enumerate(zip(path['pos'], path['color'])):
                pos = torch.tensor(pos).view(4, 2)
                pos = pos[:, [1, 0]]
                pos = (pos * 0.5 + 0.5) * self.args.image_scale

                col = torch.tensor(col.tolist() + [1.0, ])

                des_path_dict[log_idx]['pos'].append(pos.float())
                des_path_dict[log_idx]['color'].append(col.float())

        return des_path_dict
                    

    def path_dict(self):
        path_dict = dict()

        for log_idx, shapes in self.shapes_log.items():
            if log_idx == 'best':
                continue

            path_dict[log_idx] = {
                'pos': list(),
                'color': list(),
            }

            for idx, path in enumerate(shapes):
                pos = path.points / self.args.image_scale
                pos = (-pos[:, [1, 0]]) + 1
                pos = (1 - pos * 2).flatten().tolist()
                path_dict[log_idx]['pos'].append(pos)

                col = self.color_log[log_idx][idx].stroke_color[:3].tolist()
                path_dict[log_idx]['color'].append(col)
                
        return path_dict
        
    def path_dict_np(self, radius=None):
        path_dict = self.path_dict()
        for log_idx, shapes in path_dict.items():
            shapes['pos'] = np.array(shapes['pos'], dtype=np.float32)
            shapes['color'] = np.array(shapes['color'], dtype=np.float32)
            if radius is not None:
                shapes['radius'] = np.array([radius] * len(shapes['color']), dtype=np.float32)
        return path_dict


    def dino_attn(self):
        patch_size=8 # dino hyperparameter
        threshold=0.6

        # for dino model
        mean_imagenet = torch.Tensor([0.485, 0.456, 0.406])[None,:,None,None].to(self.device)
        std_imagenet = torch.Tensor([0.229, 0.224, 0.225])[None,:,None,None].to(self.device)
        totens = transforms.Compose([
            transforms.Resize((self.canvas_height, self.canvas_width)),
            transforms.ToTensor()
            ])
        
        self.main_im = Image.open(self.target_path).convert("RGB")
        main_im_tensor = totens(self.main_im).to(self.device)
        img = (main_im_tensor.unsqueeze(0) - mean_imagenet) / std_imagenet
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size
        
        with torch.no_grad():
            attn = self.dino_model.get_last_selfattention(img).detach().cpu()[0]

        nh = attn.shape[0]
        attn = attn[:,0,1:].reshape(nh,-1)
        val, idx = torch.sort(attn)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
        
        attn = attn.reshape(nh, w_featmap, h_featmap).float()
        attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
        
        return attn


    def define_attention_input(self, target_im):
        data_transforms = transforms.Compose([
                    self.clip_preprocess.transforms[-1],
                ])
        self.image_input_attn_clip = data_transforms(target_im).to(self.device)
        self.inds_image_input = target_im.squeeze(0).permute(1, 2, 0).cpu().numpy()
        

    def clip_attn(self):
        text_input = clip.tokenize([self.text_target]).to(self.device)

        if "RN" in self.saliency_clip_model:
            saliency_layer = "layer4"
            attn_map = gradCAM(
                self.clip_model.visual,
                self.image_input_attn_clip,
                self.clip_model.encode_text(text_input).float(),
                getattr(self.clip_model.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        else:
            # attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device, index=0).astype(np.float32)
            attn_map = interpret(self.image_input_attn_clip, text_input, self.clip_model, device=self.device)
            
        return attn_map

    def set_attention_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.dino_attn()
        elif self.saliency_model == "clip":
            return self.clip_attn()
        

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum() 

    def set_inds_clip(self):
        attn_map = (self.attention_map - self.attention_map.min()) / (self.attention_map.max() - self.attention_map.min())
        if self.xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(1,2,0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map
            
        def avoid_zero(x, eps=1e-8):
            return np.clip(x, eps, x.max())

        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)
        
        mask_area = (self.inds_image_input < 1.0).all(axis=-1).astype(np.float32)
        attn_map_soft = attn_map_soft * avoid_zero(mask_area)
        attn_map_soft /= attn_map_soft.sum()

        self.inds_attn_map = attn_map_soft
        
        k = self.num_stages * self.num_paths
        # try:
        #     self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False, p=attn_map_soft.flatten())
        # except ValueError:
        #     self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False)
        # self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
        
        # U, _, VT = np.linalg.svd(attn_map_soft)
        # xs = VT[:k].argmax(axis=1)
        # ys = U[:, :k].argmax(axis=0)
        # self.inds = np.stack([ys, xs], axis=1)

        # U, _, VT = np.linalg.svd(attn_map_soft)
        # inds = []
        # for i in range(k):
        #     inds.append(np.unravel_index((U[:, i:i + 1] @ VT[i:i + 1]).argmax(), attn_map_soft.shape))
        # self.inds = np.array(inds, dtype=np.int64)

        peak_height = 1.0 / k
        peak_width = 5.0
        padding_h = round(attn_map_soft.shape[0] * 0.05)
        padding_w = round(attn_map_soft.shape[1] * 0.05)
        working_attn_map = attn_map_soft.copy()
        working_attn_map[:padding_h] = -np.inf
        working_attn_map[-padding_h:] = -np.inf
        working_attn_map[:, :padding_w] = -np.inf
        working_attn_map[:, -padding_w:] = -np.inf
        inds = []
        for i in range(k):
            argmax = np.unravel_index(working_attn_map.argmax(), working_attn_map.shape)
            inds.append(argmax)
            if i < k - 1:    
                gaussian = np.zeros_like(working_attn_map)
                gaussian[argmax[0], argmax[1]] = 1.0
                gaussian = gaussian_filter(gaussian, peak_width)
                gaussian *= (peak_height / gaussian.max())
                working_attn_map -= gaussian
        self.inds = np.array(inds, dtype=np.int64)
    
        return attn_map_soft



    def set_inds_dino(self):
        k = max(3, (self.num_stages * self.num_paths) // 6 + 1) # sample top 3 three points from each attention head
        num_heads = self.attention_map.shape[0]
        self.inds = np.zeros((k * num_heads, 2))
        # "thresh" is used for visualisaiton purposes only
        thresh = torch.zeros(num_heads + 1, self.attention_map.shape[1], self.attention_map.shape[2])
        softmax = nn.Softmax(dim=1)
        for i in range(num_heads):
            # replace "self.attention_map[i]" with "self.attention_map" to get the highest values among
            # all heads. 
            topk, indices = np.unique(self.attention_map[i].numpy(), return_index=True)
            topk = topk[::-1][:k]
            cur_attn_map = self.attention_map[i].numpy()
            # prob function for uniform sampling
            prob = cur_attn_map.flatten()
            prob[prob > topk[-1]] = 1
            prob[prob <= topk[-1]] = 0
            prob = prob / prob.sum()
            thresh[i] = torch.Tensor(prob.reshape(cur_attn_map.shape))

            # choose k pixels from each head            
            inds = np.random.choice(range(cur_attn_map.flatten().shape[0]), size=k, replace=False, p=prob)
            inds = np.unravel_index(inds, cur_attn_map.shape)
            self.inds[i * k: i * k + k, 0] = inds[0]
            self.inds[i * k: i * k + k, 1] = inds[1]
        
        # for visualisaiton
        sum_attn = self.attention_map.sum(0).numpy()
        mask = np.zeros(sum_attn.shape)
        mask[thresh[:-1].sum(0) > 0] = 1
        sum_attn = sum_attn * mask
        sum_attn = sum_attn / sum_attn.sum()
        thresh[-1] = torch.Tensor(sum_attn)

        # sample num_paths from the chosen pixels.
        prob_sum = sum_attn[self.inds[:,0].astype(np.int), self.inds[:,1].astype(np.int)]
        prob_sum = prob_sum / prob_sum.sum()
        
        self.inds_image_input = np.array(self.main_im).astype(np.float32) / 255.0
        self.inds_attn_map = prob_sum

        new_inds = []
        for i in range(self.num_stages):
            new_inds.extend(np.random.choice(range(self.inds.shape[0]), size=self.num_paths, replace=False, p=prob_sum))
        self.inds = self.inds[new_inds]
        print("self.inds",self.inds.shape)
    
        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] =  self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] =  self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return thresh

    def set_attention_threshold_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.set_inds_dino()
        elif self.saliency_model == "clip":
            return self.set_inds_clip()
        

    def get_attn(self):
        return self.attention_map
    
    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds
    
    def get_mask(self):
        return self.mask

    def set_random_noise(self, epoch):
        if epoch % self.args.save_interval == 0:
            self.add_random_noise = False
        else:
            self.add_random_noise = "noise" in self.args.augemntations

class PainterOptimizer:
    def __init__(self, args, renderers):
        self.renderers = renderers
        self.points_lr = args.lr
        self.color_lr = args.color_lr
        self.args = args
        self.optim_color = args.force_sparse

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(sum([list(renderer.parameters()) for renderer in self.renderers], []), lr=self.points_lr)
        if self.optim_color:
            self.color_optim = torch.optim.Adam(sum([list(renderer.set_color_parameters()) for renderer in self.renderers], []), lr=self.color_lr)

    def update_lr(self, counter):
        new_lr = utils.get_epoch_lr(counter, self.args)
        for param_group in self.points_optim.param_groups:
            param_group["lr"] = new_lr
    
    def zero_grad_(self):
        self.points_optim.zero_grad()
        if self.optim_color:
            self.color_optim.zero_grad()
    
    def step_(self, optimize_points=True, optimize_colors=True):
        if optimize_points:
            self.points_optim.step()
        if self.optim_color and optimize_colors:
            self.color_optim.step()
    
    def get_lr(self):
        return self.points_optim.param_groups[0]['lr']


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad




def interpret(image, texts, model, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = [] # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
        cams.append(cam)  
        R = R + torch.bmm(cam, R)
              
    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam    


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma=0.98
        self.phi=200
        self.eps=-0.1
        self.sigma=0.8
        self.binarize=True
        
    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0  + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff
