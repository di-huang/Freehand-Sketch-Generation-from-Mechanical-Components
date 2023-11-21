import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize, Compose, CenterCrop, Normalize, InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

import utils.sketch_utils as sketch_utils
from models.renderer import Renderer
from models.resnet import *
from models.resnext import *
import torchvision
import third_party.CLIP_.clip as clip

from utils.shared import args
from utils.shared import stroke_config as config


CLIP_encoder = None


def linear_(in_dim, out_dim, bn=True):
    if bn:
        return [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        ]
    return [
        nn.Linear(in_dim, out_dim),
        nn.ReLU(inplace=True),
    ]

def conv_(in_channels, out_channels, kernel_size, stride, padding, bn=True):
    if bn:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
    ]
    

class StrokeGenerator(nn.Module):
    def __init__(self, patch_num, n_hidden, n_layers):
        super(StrokeGenerator, self).__init__()

        self.n_lines = config.n_lines                                               # number of real lines
        self.n_lines_decode = config.n_lines + int(config.connected)                # number of lines to be decoded
        self.n_position = config.n_pos
        self.n_decode_color = 0
        self.n_decode_rad = 0

        self.row_embed = nn.Parameter(torch.rand(patch_num[0], n_hidden // 2))
        self.col_embed = nn.Parameter(torch.rand(patch_num[1], n_hidden // 2))

        decoder_layers = torch.nn.TransformerDecoderLayer(n_hidden, 8, batch_first=True, dim_feedforward=n_hidden*2,
                                                          activation='gelu', norm_first=True, dropout=0.0)
        self.decoder_norm = nn.LayerNorm(n_hidden, eps=1e-5)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layers, n_layers, self.decoder_norm)

        self.intermediate_strokes = dict()
        self.init_stroke = nn.Parameter(torch.randn(self.n_lines_decode, n_hidden) * 0.02)

        # number of decoded parameters for a single stroke
        n_decode_params = self.n_position + self.n_decode_color + self.n_decode_rad
        if config.connected:
            n_decode_params += 2

        self.decode_stroke = nn.Sequential(
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden // 2, n_hidden // 4),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden // 4, n_decode_params),
        )

        self.init_hook()

    def init_hook(self):
        for idx, layer in enumerate(self.transformer_decoder.layers):
            layer.register_forward_hook(self.save_outputs_hook(idx+1))
            self.intermediate_strokes[idx+1] = torch.empty(0)

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            if self.decoder_norm is not None:
                self.intermediate_strokes[layer_id] = self.decoder_norm(output)
            else:
                self.intermediate_strokes[layer_id] = output
        return fn

    def decode_to_params(self, hidden):
        bs = hidden.shape[0]
        hidden = self.decode_stroke(hidden)
        device = hidden.device

        p_cont, p_col, p_rad = torch.split(hidden, [self.n_position, self.n_decode_color, self.n_decode_rad], dim=2)

        if config.connected:
            p_cont = torch.cat([p_cont[:, :-1, -2:], p_cont[:, 1:]], dim=-1)                                    # [bs, nlines, ncont+2]
            p_col = p_col[:, 1:]
            p_rad = p_rad[:, 1:]

        raw_position = p_cont.view(bs, self.n_lines, -1, 2)
        if config.line_style == 'bezier':
            coordpairs = config.coordpairs

            stacked_cont = [raw_position[:, :, coordpairs[0, 0]]]
            stacked_cont += [raw_position[:, :, coordpairs[i, -1]] for i in range(coordpairs.shape[0])]
            control_position = torch.stack(stacked_cont, dim=-2)                                                 # [batch, nlines, nsegments+1, 2]
        else:
            control_position = raw_position

        p_col = torch.zeros(bs, self.n_lines, config.n_color).to(device)

        p_rad = torch.ones(bs, self.n_lines, config.n_rad).to(device)
        n_foreground = self.n_lines - config.n_back
        p_rad[:, n_foreground:] *= 5            # thick background stroke

        return {
            'position': p_cont,
            "raw_position": raw_position, 
            "control_position": control_position, 
        }

    def forward(self, cnn_feature):
        bs = cnn_feature.shape[0]
        device = cnn_feature.device
        h, w = cnn_feature.shape[-2:]

        cnn_feature_permuted = cnn_feature.flatten(2).permute(0, 2, 1)          # NChw -> N(h*w)C

        pos_embed = torch.cat([
            self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1),
            self.row_embed[:h].unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(0)

        init_stroke = self.init_stroke.repeat(bs, 1, 1).to(device)
        
        # Set the last embedding to 1 for background strokes
        init_background = torch.zeros_like(init_stroke)
        n_foreground = self.n_lines_decode - config.n_back
        init_background[:, n_foreground:, 0] = 1
        init_stroke += init_background

        hidden = self.transformer_decoder(init_stroke, cnn_feature_permuted + pos_embed)

        strokes = self.decode_to_params(hidden)

        return strokes

    def get_intermediate_strokes(self):
        return {k: self.decode_to_params(v) for k, v in self.intermediate_strokes.items()}

class LBS(nn.Module):
    def __init__(self):
        super(LBS, self).__init__()

        n_hidden = args.n_hidden
        n_layers = args.n_layers

        self.rep_type = args.rep_type
        image_size = args.image_size
        self.train_encoder = args.train_encoder

        use_l1 = False
        self.normalize = Compose([
            Resize(image_size, interpolation=BICUBIC),
            CenterCrop(image_size),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])            

        if self.train_encoder:
            if use_l1:
                self.cnn_encoder = nn.Sequential(
                    *conv_(config.n_color, n_hidden//8, 4, 2, 1),
                    *conv_(n_hidden//8, n_hidden//4, 4, 2, 1),
                    *conv_(n_hidden//4, n_hidden//2, 4, 2, 1),
                    *conv_(n_hidden//2, n_hidden, 3, 1, 1),
                )
            else:
                # NOTE: after switching the encoder, you need to modify the 'n_hidden' parameter in the 'config/base.yaml' file.

                self.cnn_encoder = resnext18(in_channel=config.n_color, num_group=32)
                # self.cnn_encoder = resnet18(in_channel=config.n_color)
                # self.cnn_encoder = torchvision.models.efficientnet_v2_s(weights=None).features
                # self.cnn_encoder = torchvision.models.mobilenet_v3_small(weights=None).features

            H, W = image_size, image_size
            patch_num = H//8, W//8
        else:
            # use pretrained clip encoder
            global CLIP_encoder
            CLIP_encoder, _ = clip.load("ViT-B/32", args.device, jit=False)
            CLIP_encoder = CLIP_encoder.visual
            CLIP_encoder.eval()

            self.normalize = Compose([
                Resize(224, interpolation=BICUBIC),
                CenterCrop(224),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

            patch_num = 7, 7

        self.stroke_generator = StrokeGenerator(patch_num, n_hidden, n_layers)

         # number of parameters for a single parameterized strokes
        self.num_stroke_params = config.n_params

        self.reset_parameters()

    def reset_parameters(self, model=None, init_type='normal', init_gain=0.05):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
                nn.init.constant_(m.bias.data, 0.0)

        if model is None:
            model = self
        print('initialize network with %s' % init_type)
        model.apply(init_func)

    def forward(self, image):
        image = self.normalize(image)

        if self.train_encoder:
            cnn_feature = self.cnn_encoder(image)
        else:
            global CLIP_encoder
            def forward_clip(image):
                dtype = CLIP_encoder.conv1.weight.dtype

                x = image.type(dtype)
                x = CLIP_encoder.conv1(x)                               # [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)               # [*, width, grid ** 2]
                x = x.permute(0, 2, 1)                                  # [*, grid ** 2, width]
                x = torch.cat([CLIP_encoder.class_embedding.to(x.dtype) \
                               + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                               , x], dim=1)                             # [*, grid ** 2 + 1, width]
                x = x + CLIP_encoder.positional_embedding.to(x.dtype)
                x = CLIP_encoder.ln_pre(x)

                x = x.permute(1, 0, 2)  # NLD -> LND
                x = CLIP_encoder.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = CLIP_encoder.ln_post(x[:, 1:, :])
                        
                x = x.permute(0, 2, 1)
                return x
            
            with torch.no_grad():
                x = forward_clip(image)
                cnn_feature = x.view(*x.shape[:2], 7, 7)

        strokes = self.stroke_generator(cnn_feature)

        return {
            'stroke': strokes
        }


class SketchModel(nn.Module):
    def __init__(self):
        super(SketchModel, self).__init__()

        self.lbs_model = LBS()

        self.renderer = Renderer(args.image_size, min(64, args.image_size))

    def forward(self, image, sketch_type=['black']):
        lbs_output = self.lbs_model(image)
        sketch = self.renderer(lbs_output['stroke'], sketch_type)

        if isinstance(sketch_type, list) or isinstance(sketch_type, tuple):
            for idx, types in enumerate(sketch_type):
                lbs_output[f'sketch_{types}'] = sketch[idx]
        else:
            lbs_output[f'sketch_{sketch_type}'] = sketch

        return lbs_output
    
    def get_intermediate_strokes(self):
        return self.lbs_model.stroke_generator.get_intermediate_strokes()

    def set_progress(self, progress):
        self.renderer.set_sigma2(progress)

    def parameters(self):
        return self.lbs_model.parameters()

