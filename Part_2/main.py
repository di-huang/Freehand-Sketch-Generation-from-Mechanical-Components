import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import get_dataset
from models.LBS import SketchModel
from models.clip_loss import Loss as CLIPLoss
from loss import LBS_loss_fn

import argparser
from utils.sketch_utils import *
from utils.shared import args, logger, update_args, update_config
from utils.shared import stroke_config as config

import warnings
from utils.my_utils import *


warnings.filterwarnings("ignore")

clip_loss_fn = None


def unpack_dataloader(datas):
    img, pos, path = datas

    return {
        "img": img.to(device, non_blocking=True),
        "pos": pos.to(device, non_blocking=True),                                           # [bs, 9, nL, 8]
        "path": path
    }


# S--------------------------------------------------
import svgwrite

def tensor2image(tensor, out="my_output_image.jpg"):
    tensor_to_pil = transforms.ToPILImage()
    image = tensor_to_pil(tensor)
    image.save(out)

def tensor_to_svg(tensor, filename='output.svg'):
    dwg = svgwrite.Drawing(filename, profile='full', size=("200px", "200px"))

    dwg.attribs['xmlns:ev'] = "http://www.w3.org/2001/xml-events"
    dwg.attribs['xmlns:xlink'] = "http://www.w3.org/1999/xlink"
    dwg.attribs['baseProfile'] = "full"
    dwg.attribs['height'] = "200"
    dwg.attribs['width'] = "200"
    dwg.attribs['viewBox'] = "0 0 10 10"
    dwg.attribs['version'] = "1.1"

    g = dwg.g(transform="rotate(-90 5 5) scale(-1 1) translate(-10 0)")

    for sketch in tensor:
        sketch = [sketch[i].cpu().item() + 3 for i in range(len(sketch))]

        path_data = "M {} {} C {} {} {} {} {} {}".format(sketch[0], sketch[1],
                                                         sketch[2], sketch[3],
                                                         sketch[4], sketch[5],
                                                         sketch[6], sketch[7])
        g.add(dwg.path(d=path_data, fill="none", stroke="black", 
                       stroke_linecap="round", stroke_linejoin="round", 
                       stroke_opacity=1.0, stroke_width=0.05))

    dwg.add(g)

    dwg.save(pretty=True)

    
def save_outputs(inputs, img_outputs, svg_outputs, root):
    if not os.path.exists(root):
        os.makedirs(root)
    # print("len(inputs):", len(inputs['path']))
    for i in range(len(inputs['path'])):
        p = inputs['path'][i]
        img_save_path = os.path.join(root, 'img_' + os.path.splitext(os.path.basename(p))[0] + '.jpg')
        svg_save_path = os.path.join(root, 'svg_' + os.path.splitext(os.path.basename(p))[0] + '.svg')
        
        tensor_to_svg(svg_outputs['stroke']['position'][i], svg_save_path)
        tensor2image(img_outputs['sketch_black'][i], img_save_path)


import matplotlib.pyplot as plt
import numpy as np

def plot_loss_report(arrays, names, title):
    plt.figure(figsize=[16,12])

    for array in arrays:
        x = [point[0] for point in array]
        y = [point[1].cpu().item() for point in array]  # Assuming this is needed for PyTorch tensors
        plt.plot(x, y, marker='o')  # Plot the line with 'o' as the marker for points
        for i, value in enumerate(y):
            plt.text(x[i], y[i], f'{value:.2f}', color = 'black', ha = 'center', va = 'bottom')

    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title(title, fontsize=22)
    plt.legend([names[i] for i in range(len(names))], fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join('logs', args.dataset, f'{title}.png'))
    plt.close()

# E--------------------------------------------------


def train(model, optimizer, scheduler, loaders):
    """Train
    Args:
        model ([type]): [description]
        optimizer ([type]): [description]
        scheduler ([type]): [description]
        loaders ([type]): [description]
    """
    train_loader, val_loader, test_loader = loaders

    global clip_loss_fn
    clip_loss_fn = CLIPLoss()

    best_loss = 1e8

    # for in range [1 ~ args.epoch], while epoch 0 is only for visualizing the initialized model.
    logger.log(f"----------args.epochs: {args.epochs}")
    logger.log(f"----------args.batch: {args.batch}")
    logger.log(f"----------args.train_encoder: {args.train_encoder}")
    logger.log(f"----------args.prev_weight: {args.prev_weight}")

    train_total_loss_list = []
    train_hf_loss_list = []
    train_percept_loss_list = []
    train_gt_loss_list = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        progress = epoch / args.epochs
        model.set_progress(progress)

        ### train
        if epoch != args.start_epoch:
            model.train()
            imgs, sketches, lbs_output_, loss_dict_ = train_epoch(model, optimizer, scheduler, train_loader, epoch) # imgs is inputs
            
            plot_results_gt(imgs, sketches, os.path.join('logs', args.dataset, 'train_log.jpg'))

            if epoch != 0:
                if epoch % 20 == 0 or epoch == 1 or epoch == args.epochs:
                    train_total_loss_list.append([epoch, loss_dict_['loss_total']])
                    train_hf_loss_list.append([epoch, loss_dict_['loss_hausdorff']])
                    train_percept_loss_list.append([epoch, loss_dict_['loss_percept']])
                    train_gt_loss_list.append([epoch, loss_dict_['loss_gt_pos']])
                    plot_loss_report([train_total_loss_list, train_hf_loss_list, train_percept_loss_list, train_gt_loss_list],\
                                     ['total', 'hausdorff', 'percept', 'guide'], 'train_loss')
                if epoch % 25 == 0 or epoch == args.epochs:
                    folder_ = os.path.join('logs', args.dataset, "seen", str(epoch))
                    if not os.path.exists(folder_):
                        os.makedirs(folder_)
                    save_outputs(imgs, sketches, lbs_output_, folder_)
                    plot_results_gt(imgs, sketches, os.path.join(folder_, 'train_log.jpg'))


        ### validation
        model.eval()
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            val_loss, imgs, sketches, lbs_output_ = validation(model, optimizer, val_loader)

            loss_gt = val_loss["loss_gt_pos"]
            loss_percept = val_loss["loss_percept"]
            loss_hf = val_loss["loss_hausdorff"]
            loss_total = val_loss["loss_total"]
            logger.log(f"[Epoch {epoch:3d}] Val --- loss_gt: {loss_gt.item():.3f} | loss_percept: {loss_percept.item():.3f} | loss_hf: {loss_hf.item():.3f} | loss_total: {loss_total.item():.3f}")

            if val_loss["loss_total"] < best_loss:
                best_loss = val_loss["loss_total"].item()
                state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(logger.basepath, 'model_best.pt'))

            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(logger.basepath, 'model.pt'))
            torch.save({
                "epoch": epoch,
                "optim": optimizer.state_dict(),
            }, os.path.join(logger.basepath, 'optim.pt'))

            plot_results_gt(imgs, sketches, os.path.join('logs', args.dataset, 'val_log.jpg'))

            ### test
            if args.testset_log:
                best_weight_path = os.path.join(logger.basepath, 'model_best.pt')
                weight_path = os.path.join(logger.basepath, 'model.pt')
                checkpoint = torch.load(best_weight_path) if os.path.exists(best_weight_path) else torch.load(weight_path)
                tmp_model = SketchModel()
                tmp_model = tmp_model.to(device)
                tmp_model.load_state_dict(checkpoint)
                tmp_model.eval()
                for idx, datas in enumerate(test_loader):
                    if idx >= 20:  # test for 20 steps
                        break
                    imgs_ = datas[0].to(device, non_blocking=True)
                    with torch.no_grad():
                        lbs_output = tmp_model(imgs_)
                    imgs = {"img": imgs_, "path": datas[1]}
                plot_results_gt(imgs, lbs_output, f'logs/{args.dataset}/test_log.jpg')
                if epoch != 0:
                    if epoch % 25 == 0 or epoch == args.epochs:
                        folder_ = os.path.join('logs', args.dataset, "unseen", str(epoch))
                        if not os.path.exists(folder_):
                            os.makedirs(folder_)
                        save_outputs(imgs, lbs_output, lbs_output, folder_)
                        plot_results_gt(imgs, lbs_output, os.path.join(folder_, 'test_log.jpg'))

    
    # plot the result using the best model weight
    weight_path = logger.basepath + "/model_best.pt"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        model.eval()
        _, imgs, sketches, lbs_output_ = validation(model, optimizer, val_loader)
        plot_results_gt(imgs, sketches, f'logs/{args.dataset}/best.jpg')
    
    logger.log(
        f"Best Val loss: {best_loss:.3f}"
    )


def plot_results_gt(inputs, sketches, output_filename='logs/my_data/log.jpg'):
    my_stacked_results = torch.stack([
        inputs["img"], sketches["sketch_black"]
    ], dim=1).flatten(0, 1)

    my_save_plots(my_stacked_results, output_filename)


def train_epoch(model, optimizer, scheduler, train_loader, epoch):
    loss_dict = {}
    for idx, datas in enumerate(train_loader):
        steps = epoch * len(train_loader) + idx

        inputs = unpack_dataloader(datas)

        sketches, loss, lbs_output_ = LBS_loss_fn(model, optimizer, clip_loss_fn, inputs, train_model=True)
        loss_dict.update(loss)

        loss_dict["lr"] = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if idx % args.print_every == 0:
            loss_gt = loss["loss_gt_pos"]
            loss_percept = loss["loss_percept"]
            loss_hf = loss["loss_hausdorff"]
            loss_total = loss["loss_total"]
            logger.log(f"[Epoch {epoch:3d} iter {idx:4d}] Train --- loss_gt: {loss_gt.item():.3f} | loss_percept: {loss_percept.item():.3f} | loss_hf: {loss_hf.item():.3f} | loss_total: {loss_total.item():.3f}")

            for name, values in loss_dict.items():
                logger.scalar_summary(name, values, steps)

    return inputs, sketches, lbs_output_, loss_dict

def validation(model, optimizer, val_loader):
    loss_total, loss_hf, loss_percept, loss_gt = 0, 0, 0, 0

    for idx, datas in enumerate(val_loader):
        if idx == 20:  # validate for 20 steps
            break

        inputs = unpack_dataloader(datas)
        with torch.no_grad():
            sketches, val_losses, lbs_output_ = LBS_loss_fn(model, optimizer, clip_loss_fn, inputs, train_model=False)
            loss_total += val_losses["loss_total"]
            loss_hf += val_losses["loss_hausdorff"]
            loss_gt += val_losses["loss_gt_pos"]
            loss_percept += val_losses["loss_percept"]

    loss_total /= (idx + 1)
    loss_hf /= (idx + 1)
    loss_gt /= (idx + 1)
    loss_percept /= (idx + 1)

    val_loss = {
        "loss_hausdorff": loss_hf,
        "loss_gt_pos": loss_gt,
        "loss_percept": loss_percept,
        "loss_total": loss_total,
    }

    return val_loss, inputs, sketches, lbs_output_


def test(test_loader, weight_path):
    checkpoint = torch.load(weight_path)
    tmp_model = SketchModel()
    tmp_model = tmp_model.to(device)
    tmp_model.load_state_dict(checkpoint)
    tmp_model.eval()
    test_result_folder = os.path.join('logs', args.dataset, "test_outputs")
    if not os.path.exists(test_result_folder):
        os.makedirs(test_result_folder)
    for idx, data in enumerate(test_loader):
        img = data[0].to(device, non_blocking=True)
        # img_path = data[1][0]
        with torch.no_grad():
            output = tmp_model(img)
        # first_slash_index = img_path.find('/')
        # result_path = os.path.join('logs', args.dataset, img_path[first_slash_index + 1:])
        # directory, _ = os.path.split(result_path)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # tensor2image(output['sketch_black'].squeeze(), os.path.join(test_result_folder, f"test_result_{idx}.jpg"))

        tensor2image(output['sketch_black'].squeeze(), os.path.join(test_result_folder, 'img_' + os.path.splitext(os.path.basename(data[1][0]))[0] + '.jpg'))
        tensor_to_svg(output['stroke']['position'][0], os.path.join(test_result_folder, 'svg_' + os.path.splitext(os.path.basename(data[1][0]))[0] + '.svg'))

import shutil

def main():
    if os.path.exists('seen'):
        shutil.rmtree('seen')
    if os.path.exists('unseen'):
        shutil.rmtree('unseen')

    args_ = argparser.parse_arguments()

    train_set, val_set, test_set, image_shape = get_dataset(args_)

    args_.image_size = image_shape[1]
    args_.image_num_channel = image_shape[0]
    stroke_config = argparser.get_stroke_config(args_)

    update_args(args_)
    update_config(stroke_config)

    global device
    device = args.device

    train_loader = DataLoader(train_set, shuffle=True, num_workers=16, pin_memory=True, batch_size=args.batch)
    val_loader = DataLoader(val_set, shuffle=True, num_workers=8, pin_memory=True, batch_size=args.batch)
    test_loader = DataLoader(test_set, shuffle=True, num_workers=8, pin_memory=True, batch_size=64)

    model = SketchModel()
    model = model.to(device)

    t_0 = args.epochs * len(train_loader)
    optimizer = optim.AdamW(model.parameters(), lr=0, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, eta_max=args.lr, T_0=t_0, T_mult=1, T_up=t_0 // 20, gamma=0.5)

    if args.load_path is not None:
        checkpoint = torch.load(os.path.join(args.load_path, "model.pt"))
        model.load_state_dict(checkpoint)
        checkpoint_opt = torch.load(os.path.join(args.load_path, "optim.pt"))["optim"]
        optimizer.load_state_dict(checkpoint_opt)
        args.start_epoch = torch.load(os.path.join(args.load_path, "optim.pt"))["epoch"]
        scheduler.step(args.start_epoch * len(train_loader))

    xp_time = time.strftime("%m%d-%H%M%S")
    logger.init(
        xpid=args.xpid,
        tag=f"{args.comment}_seed{args.seed}",
        xp_args=args.__dict__,
        rootdir="logs",
        timestamp=xp_time,
        use_tensorboard=(not args.no_tensorboard),
        resume=False,
    )
    args.logdir = logger.basepath

    logger.log(model)
    logger.log(f"# Params: {count_parameters(model)}")
    args.starting_step = 0

# S ------------------------------------------------
    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=(train_loader, val_loader, test_loader)
    )

    weight_root_path = args.test_weight_path if args.test_weight_path else logger.basepath
    test_loader = DataLoader(test_set, shuffle=True, num_workers=8, pin_memory=True, batch_size=1)
    best_weight_path = os.path.join(weight_root_path, 'model_best.pt')
    weight_path = best_weight_path if os.path.exists(best_weight_path) else os.path.join(weight_root_path, 'model.pt')
    test(test_loader, weight_path)
# E ------------------------------------------------


if __name__ == "__main__":
    main()

