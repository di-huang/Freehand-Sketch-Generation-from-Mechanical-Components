import argparse
import os
import random
import numpy as np
import torch
from utils.config import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data", help="dataset root directory")
    parser.add_argument("--no_cuda", action='store_true', help="load model in cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config_path", type=str, default="config/clevr.yaml")

    # =================================
    # ======== strokes config =========
    # =================================
    parser.add_argument("--num_strokes", type=int, default=20, help="number of strokes (including background)")
    parser.add_argument("--num_background", type=int, default=0, help="number of background strokes")    
    parser.add_argument("--num_segments", type=int, default=1,
                        help="number of segments for each stroke, each stroke is a bezier curve with 4 control points")
    parser.add_argument("--saliency_clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--line_style", type=str, default="bezier", choices=["bezier", "line", "point"])
    parser.add_argument("--radius", type=float, default=0.0005)
    parser.add_argument("--power_i", type=float, default=1.0)
    parser.add_argument("--power_f", type=float, default=2.0)
    parser.add_argument("--smooth_segment", action='store_true')
    parser.add_argument("--connected", action='store_true')


    # =================================
    # =========== CLIP loss ===========
    # =================================
    parser.add_argument("--num_aug_clip", type=int, default=3) # LBS: 4
    parser.add_argument("--clip_conv_loss_type", type=str, default="L2")
    parser.add_argument("--clip_conv_layer_weights",
                        type=str, default="0,0,1.0,1.0,0")
    parser.add_argument("--clip_model_name", type=str, default="RN101")
    parser.add_argument("--clip_fc_loss_weight", type=float, default=0.1)
    

    # =================================
    # ======== Hyperparameters ========
    # =================================
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--lbd_g', default=1.0, type=float, help='weight for L_guide')
    parser.add_argument('--lbd_p', default=1.0, type=float, help='weight for L_percept')
    parser.add_argument('--lbd_e', default=1.0, type=float, help='weight for L_embed')
    parser.add_argument('--lbd_h', default=0.1, type=float, help='weight for L_hausdorff')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--dataset', type=str, default='my_data')

    parser.add_argument('--n_hidden', default=512, type=int)
    parser.add_argument('--n_embedding', default=512, type=int)
    parser.add_argument('--d_project', default=128, type=int) # unused
    parser.add_argument('--n_layers', default=8, type=int)

    parser.add_argument('--hungarian', action='store_true')
    parser.add_argument('--prev_weight', type=float, default=0.0)
    parser.add_argument('--embed_loss', type=str, choices=['none', 'ce', 'simclr', 'supcon'], default='none')
    parser.add_argument('--train_encoder', action='store_true')
    parser.add_argument('--rep_type', type=str, default='LBS+')

    # =================================
    # ========== MoCo Options =========
    # =================================
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--max_queue', type=int, default=4096)
    parser.add_argument('--momentum', type=int, default=0.999)

    # =================================
    # ============ logging ============
    # =================================
    parser.add_argument('--print_every', help='', default=50, type=int)
    parser.add_argument('--evaluate_every', help='', default=10, type=int)
    parser.add_argument('--validate_every', help='', default=1, type=int)
    parser.add_argument('--comment', help='Comment', default='test', type=str)
    parser.add_argument('--xpid', help='Distinguishable experiment id', default='test', type=str)
    parser.add_argument('--no_tensorboard', help='Disable Tensorboard SummaryWriter', action='store_true')
    parser.add_argument('--no_eval', help='no evaluation', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--load_path', type=str, default=None)

    # =================================
    # ========== evaluation ===========
    # =================================
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_epochs", type=int, default=100)
    parser.add_argument("--eval_lr_cand", nargs=2, type=float, default=[1, 0.1])

    # optimization
    parser.add_argument('--eval_lr_decay_epochs', type=str, default='60,75,90', help='where to decay lr, can be a list')
    parser.add_argument('--eval_lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--eval_weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--eval_momentum', type=float, default=0.9, help='momentum')

    # other setting
    parser.add_argument('--eval_cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--eval_warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--eval_regression', action='store_true')
    parser.add_argument('--eval_angle', action='store_true')
    parser.add_argument("--custom_transforms_size", type=int, default=128)
    parser.add_argument('--key_steps', type=int, nargs='+',
                        default=[0, 50, 100, 200, 400, 700, 1000, 1500, 2000])
    parser.add_argument('--label_data_source', type=str, default='pkl', help='init_clipasso or pkl')
    parser.add_argument("--train_val_split_ratio", type=float, default=0)
    parser.add_argument("--testset_log", type=int, default=1)
    parser.add_argument("--test_weight_path", type=str, default='')


    # set default arguments from parser
    args_ = vars(parser.parse_known_args()[0])
    args = Config.from_dict(args_)

    # set base, specified configurations
    args.update(Config.from_yaml('config/base.yaml'))
    args.update(Config.from_yaml(args.config_path))
    
    # set specified arguments in command line
    aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    for arg in ['disable_radius', 'disable_color']:
        aux_parser.add_argument('--'+arg, action='store_false', dest=arg.replace('disable', 'enable'))
    for arg in args_: 
        if type(args_[arg]) == bool:
            aux_parser.add_argument('--'+arg, action='store_true')
        else:
            aux_parser.add_argument('--'+arg, type=type(args_[arg]))
    cli_args, _ = aux_parser.parse_known_args()
    args.update(Config.from_dict(vars(cli_args)))
    
    # set seed
    set_seed(args.seed)

    args.clip_conv_layer_weights = [
        float(item) for item in args.clip_conv_layer_weights.split(',')]
    
    if args.embed_loss == 'none':
        args.lbd_e = 0.0
        if args.rep_type == 'LBS+':     # disable z_e for final representation
            args.rep_type = 'LBS'
    
    if not args.no_cuda:
        args.device = torch.device(
            "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
        )
    else:
        args.device = torch.device("cpu")
    return args


def get_stroke_config(args):

    assert args.line_style in ['bezier', 'line', 'point']

    if args.line_style == 'bezier':
        if args.smooth_segment:
            npoints = (4 + (args.num_segments - 1) * 2)  # 4 points + 2 for each intermediate [pt, cp]

            # this is a list of pairs of "connections" 0-1-2-3, 3-2-4-5, 5-4-6-7, ...
            coordpairs = torch.stack([torch.arange(0, npoints - 3, 2),
                                      torch.arange(1, npoints - 2, 2),
                                      torch.arange(2, npoints - 1, 2),
                                      torch.arange(3, npoints - 0, 2)], dim=1)
            coordpairs[1:, 0] += 1
            coordpairs[1:, 1] -= 1
        else:
            npoints = (4 + (args.num_segments - 1) * 3)  # 4 points + 3 for each intermediate [pt, cp1, cp2]

            # this is a list of pairs of "connections" 0-1-2-3, 3-4-5-6, 6-7-8-9, ...
            coordpairs = torch.stack([torch.arange(0, npoints - 3, 3),
                                      torch.arange(1, npoints - 2, 3),
                                      torch.arange(2, npoints - 1, 3),
                                      torch.arange(3, npoints - 0, 3)], dim=1)
        
        if args.connected:
            npoints -= 1

    elif args.line_style == 'line':
        npoints = args.num_segments + 1
        coordpairs = torch.stack([torch.arange(0, npoints - 1, 1), torch.arange(1, npoints, 1)], dim=1)
        
        if args.connected:
            npoints -= 1

    elif args.line_style == 'point':
        args.num_segments = 1
        npoints = 1
        coordpairs = torch.arange(0, npoints, 1).unsqueeze(1)
        args.connected = False

    n_positions, n_radius, n_colors = 2 * npoints, 1, args.image_num_channel
    n_params = n_positions + n_radius + n_colors

    return Config.from_dict({
        "n_pos": n_positions,
        "n_rad": n_radius,
        "n_color": n_colors,
        "n_back": args.num_background,
        "n_lines": args.num_strokes,
        "n_points": npoints,
        "n_segments": args.num_segments,
        "coordpairs": coordpairs,
        "radius": args.radius,
        "power_i": args.power_i,
        "power_f": args.power_f,
        "enable_b": False,
        "smooth": args.smooth_segment,
        "line_style": args.line_style,
        "connected": args.connected,
        "n_params": n_params
    })
