import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from hausdorff import hausdorff_distance

from utils.sketch_utils import *
from utils.shared import args, logger
from utils.shared import stroke_config as config


criterion_l1 = torch.nn.L1Loss().cuda()


def hungarian_match(position_g, valid_g, position_s):
    bs, nL = position_s.shape[:2]
    cur_valid_gt_size = 0

    with torch.no_grad():
        r_idx = []
        c_idx = []
        for i in range(position_g.shape[0]):
            is_valid_gt = valid_g[i]
            cost_matrix_l1 = torch.cdist(position_s[i], position_g[i, is_valid_gt], p=1)        # [nL, nvalid]
            r, c = linear_sum_assignment(cost_matrix_l1.cpu())                                  # [npair], [npair]
            r_idx.append(torch.tensor(r + nL * i).cuda())
            c_idx.append(torch.tensor(c + cur_valid_gt_size).cuda())
            cur_valid_gt_size += is_valid_gt.int().sum().item()

        r_idx = torch.cat(r_idx, dim=0)                                                         # [Npair]
        c_idx = torch.cat(c_idx, dim=0)                                                         # [Npair]
        paired_gt_decision = torch.zeros(bs * nL).cuda()                                        # [bs * nL]
        paired_gt_decision[r_idx] = 1.0

    return r_idx, c_idx


def hungarian_loss(stroke, gt):
    position_g = gt["position"]                         # [bs, nL, npos]
    valid_g = position_g.abs().mean(dim=2) < 0.95

    position_s = stroke["position"][:, :config.n_lines]        # [bs, nL, npos]

    assert position_g.shape == position_s.shape
    r_idx, c_idx = hungarian_match(position_g, valid_g, position_s)

    paired_gt_param = position_g[valid_g][c_idx, :]                                             # [Npair, npos]
    paired_pred_param = position_s.flatten(end_dim=1)[r_idx, :]                                 # [Npair, npos]

    loss_gt = criterion_l1(paired_pred_param, paired_gt_param.detach())

    return loss_gt


def update_model(model, opt, loss):
    model.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    opt.step()


def guide_loss_fn(inputs, lbs_output):
    stroke = lbs_output['stroke']
    intermediate = lbs_output['intermediate']
    gt = {"position": inputs["pos"][:, -1]}
    loss_gt = hungarian_loss(stroke, gt)

    ### progressive optimization process
    # print('args.n_layers:', args.n_layers) # depends on argparser
    # print('len(intermediate):', len(intermediate), '; type(intermediate):', type(intermediate)) # depends on # of transformer decoder layers
    # print('inputs["pos"].shape:', inputs["pos"].shape) # depends on # of clipasso iterations
    # assert False
    if args.prev_weight > 0:
        for layer_idx in range(1, args.n_layers):
            pos = inputs["pos"][:, layer_idx]
            stroke = intermediate[layer_idx]
            gt = {"position": pos}
            loss_gt_ = hungarian_loss(stroke, gt)
            loss_gt += loss_gt_ * args.prev_weight

    return loss_gt


def hausdorff_loss(stroke, gt):
    position_g = gt["position"]
    position_s = stroke["position"][:, :config.n_lines]

    position_s = position_s.cpu().detach().numpy()
    position_g = position_g.cpu().detach().numpy()

    loss = 0
    for i in range(position_g.shape[0]):
        loss += hausdorff_distance(position_s[i], position_g[i], distance='manhattan')
    loss /= int(position_g.shape[0])

    return loss


def hausdorff_loss_fn(inputs, lbs_output):
    stroke = lbs_output['stroke']
    intermediate = lbs_output['intermediate']
    gt = {"position": inputs["pos"][:, -1]}
    loss = hausdorff_loss(stroke, gt)

    ### progressive optimization process
    if args.prev_weight > 0:
        for layer_idx in range(1, args.n_layers):
            stroke = intermediate[layer_idx]
            gt = {"position": inputs["pos"][:, layer_idx]}
            loss_ = hausdorff_loss(stroke, gt)
            loss += loss_ * args.prev_weight

        loss /= int(args.n_layers)

    return torch.tensor(loss, requires_grad=False, dtype=torch.float32).to(args.device)


def LBS_loss_fn(model, opt, clip_loss_fn, inputs, train_model=True):
    img = inputs["img"]

    lbs_output = model(img)
    lbs_output['intermediate'] = model.get_intermediate_strokes()
    sketch_black = lbs_output['sketch_black']

# S ------------------------------------------------

    ##### L_{local} #####
    loss_gt = guide_loss_fn(inputs, lbs_output)
    loss_gt *= args.lbd_g

    ##### L_{global} #####
    if args.lbd_h > 0:
        loss_hf = hausdorff_loss_fn(inputs, lbs_output)
        loss_hf *= args.lbd_h
    else:
        loss_hf = torch.zeros(1).to(args.device)

    ##### L_{clip-based} #####
    if args.lbd_p != 0:
        clip_loss_dict = clip_loss_fn(sketch_black, img, None, None, 1, None)
        loss_percept = sum(list(clip_loss_dict.values())) * args.lbd_p
    else:
        loss_percept = torch.zeros(1).to(args.device)

    loss_hf = loss_hf * 8
    # loss_percept = loss_percept * 15
    # loss_total = loss_gt + loss_percept
    loss_total = loss_gt + loss_percept + loss_hf
    # loss_total = loss_gt + loss_hf

    if train_model:
        update_model(model, opt, loss_total)

    losses = {
        "loss_hausdorff": loss_hf,
        "loss_gt_pos": loss_gt,
        "loss_percept": loss_percept,
        "loss_total": loss_total,
    }

# E ------------------------------------------------

    return {
        "input_images": img,
        "sketch_black": sketch_black
    }, losses, lbs_output

