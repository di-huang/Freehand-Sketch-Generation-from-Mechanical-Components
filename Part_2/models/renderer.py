import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from third_party.disttrans import *
from third_party.raster import exp
from third_party.composite import *

from utils.shared import args
from utils.shared import stroke_config as config


class Renderer(nn.Module):
    def __init__(self, image_size, canvas_size):
        """Initialize Differentiable Rasterizer.

        Args:
            image_size (int): Resolution of the image to generate
            canvas_size (int): Resolution of the grid to actually rasterize
            stroke_config (dict): Stroke configuration
            device (str, optional): Device. Defaults to 'cuda:0'.
        """
        super(Renderer, self).__init__()

        self.image_size = image_size
        self.device = args.device
        self.canvas_size = canvas_size
        if image_size == canvas_size:
            self.resize = lambda x: x
        else:
            self.resize = Resize(image_size)

        r = torch.linspace(-1, 1, canvas_size)
        c = torch.linspace(-1, 1, canvas_size)
        grid = torch.meshgrid(r, c)
        self.grid = torch.stack(grid, dim=2).to(self.device)

        self.set_sigma2(0.0)

    def set_sigma2(self, progress):
        """Anneals the sharpness of the rasterized stroke by the training progress.

        Args:
            progress (float): The overall progress in training, a value between 0 and 1.
        """
        ratio = config.power_f / config.power_i
        self.power = config.power_i * (ratio**progress)
        self.sigma2 = ((1 - ratio) * progress + ratio) * config.radius

    def reshape_params(self, stroke, background="keep"):
        assert background in ["keep", "remove", "only"]
        # the latent_to_linecoord process will map the input latent vector to control points
        bs = stroke["position"].shape[0]
        device = self.device
        n_back = config.n_back

        if background == "keep":
            nlines = config.n_lines
            idx_i = 0
            idx_f = nlines
        elif background == "remove":
            nlines = config.n_lines - n_back
            idx_i = 0
            idx_f = nlines
        elif background == "only":
            nlines = n_back
            if n_back == 0:
                idx_i = config.n_lines
            else:
                idx_i = -n_back
            idx_f = config.n_lines

        nsegments = config.n_segments
        coordpairs = config.coordpairs
        nc = config.n_color

        p_pos = stroke["position"][:, idx_i:idx_f]

        if "radius" not in stroke:  # use default thickness (1^2 for foreground, 5^2 for background)
            p_rad = torch.ones(bs, nlines, 1).to(device)
            if background == "keep":
                if n_back != 0:
                    p_rad[:, -n_back:] *= 25
            elif background == "only":
                p_rad *= 25
        else:
            p_rad = stroke["radius"][:, idx_i:idx_f] ** 2

        if "color" not in stroke:  # use default color (black)
            p_col = torch.zeros(bs, nlines, nc).to(device)
        else:
            p_col = stroke["color"][:, idx_i:idx_f]

        if "background" not in stroke:  # use default color (white)
            p_back = torch.ones(bs, nc).to(device)
        else:
            p_back = stroke["background"]

        lines = p_pos.view(bs, nlines, -1, 2)                                                           # [batch, nlines, npoints, 2]
        stacked_pos = [lines[:, :, coordpairs[:, i]] for i in range(coordpairs.shape[1])]
        lines_ = torch.stack(stacked_pos, dim=-2)                                                       # [batch, nlines, nsegments, ncoords, 2]
        if config.smooth and config.line_style == "bezier":
            # reflect control point vectors
            lines_[:, :, 1:, 1] = 2 * lines_[:, :, 1:, 0] - lines_[:, :, 1:, 1]

        lines_ = lines_.view(bs, nlines * nsegments, -1, 2)                                             # [batch, nlines*nsegments, ncoords, 2]
        sigma = p_rad.view(bs, -1, 1, 1).repeat_interleave(nsegments, dim=1) * self.sigma2              # [batch, nlines*nsegments, 1, 1]
        color = p_col.view(bs, nlines, nc, 1, 1).repeat_interleave(nsegments, dim=1)                    # [batch, nlines*nsegments, 3, 1, 1]
        background = p_back.view(bs, nc, 1, 1).expand(-1, -1, *self.grid.shape[:2])                     # [batch, 3, r, c]

        return lines_, sigma, color, background

    def create_euclidean_distance_square(self, lines):
        if config.line_style == "line":
            edt2 = line_edt2(lines, self.grid)

        elif config.line_style == "bezier":
            edt2 = curve_edt2_polyline(lines, self.grid, 10)

        elif config.line_style == "point":
            edt2 = point_edt2(lines.squeeze(2), self.grid)

        else:
            raise NotImplementedError(f"line style {config.line_style} not implemented")

        return edt2

    def rendering_reshaped_params(self, params, canvas=None):
        lines, sigma, color, background = params

        # Will overlay strokes on top of canvas and will optionally use the background color instead of canvas.
        if config.enable_b or canvas is None:
            canvas = background

        canvas = canvas.unsqueeze(1)

        distance_square = self.create_euclidean_distance_square(lines)                                  # [bs, nlines*nsegments, row, col]
        intensity = exp(distance_square, sigma, self.power)

        intensity = intensity.unsqueeze(2).repeat_interleave(config.n_color, dim=2)                     # [bs, nlines*nsegments, 3, row, col]
        rasters = color * intensity
        rasters = torch.cat([rasters, canvas], dim=1)

        # intensity without color
        intensity_ = torch.cat([intensity, torch.ones_like(canvas)], dim=1)
        linv = 1 - intensity_ + 1e-10
        images = over(rasters, linv, dim=1, keepdim=False)                                              # [bs, 3, row, col]

        return self.resize(images).clip(0, 1)

    def render_full(self, stroke):
        reshaped = self.reshape_params(stroke, background="keep")
        return self.rendering_reshaped_params(reshaped)

    def render_foreground(self, stroke, colored=True):
        if config.n_lines == config.n_back:
            return self.white_canvas(stroke["position"].shape[0])

        reshaped = self.reshape_params(stroke, background="remove")
        p, s, c, b = reshaped
        if not colored:
            c_ = torch.zeros_like(c)
            b_ = torch.ones_like(b)
            reshaped = p, s, c_, b_
        return self.rendering_reshaped_params(reshaped)
    
    def render_background(self, stroke):
        if config.n_back == 0:
            return self.white_canvas(stroke["position"].shape[0])

        reshaped = self.reshape_params(stroke, background="only")
        return self.rendering_reshaped_params(reshaped)

    def render_each_stroke(self, stroke):
        for key in stroke.keys():
            if stroke[key] is not None:
                stroke[key] = stroke[key].detach()

        results = []
        import copy
        for idx in range(config.n_lines - config.n_back):
            params_ = copy.deepcopy(stroke)
            c = params_["color"][:, idx]
            params_["color"] = 0.8 + params_["color"] * 0.2
            params_["color"][:, idx] = c

            permute = torch.arange(0, config.n_lines)
            permute[: idx + 1] -= 1
            permute[0] = idx

            params_["color"] = params_["color"][:, permute]
            params_["position"] = params_["position"][:, permute]

            reshaped = self.reshape_params(params_, background="remove")
            results.append(self.rendering_reshaped_params(reshaped))

        return results
    
    def white_canvas(self, batch_size):
        return torch.ones((batch_size, config.n_color, self.image_size, self.image_size)).to(self.device)
    
    def forward(self, stroke, sketch_type=["color", "black", "background", "foreground"]):
        sketch_fn = {
            "color": self.render_full,
            "black": lambda x: self.render_foreground(x, colored=False),
            "foreground": lambda x: self.render_foreground(x, colored=True),
            "background": self.render_background,
            "single": self.render_each_stroke,
        }

        if isinstance(sketch_type, list) or isinstance(sketch_type, tuple):
            return [sketch_fn[type_](stroke) for type_ in sketch_type]
        
        return sketch_fn[sketch_type](stroke)