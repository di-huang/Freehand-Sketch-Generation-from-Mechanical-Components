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
    img, pos = datas

    return {
        "img": img.to(device, non_blocking=True),
        "pos": pos.to(device, non_blocking=True)                                           # [bs, 9, nL, 8]
    }


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
    for epoch in range(args.start_epoch, args.epochs + 1):
        progress = epoch / args.epochs
        model.set_progress(progress)

        ### train
        if epoch != args.start_epoch:
            model.train()
            imgs, sketches = train_epoch(model, optimizer, scheduler, train_loader, epoch)
            plot_results_gt(imgs, sketches, os.path.join('logs', args.dataset, 'train_log.jpg'))

        ### validation
        model.eval()
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            val_loss, imgs, sketches = validation(model, optimizer, val_loader)

            logger.log(
                f"[Epoch {epoch:3d}] \t\t Val loss: {val_loss.item():.3f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss.item()
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
                    datas = datas[0]
                    imgs_ = datas.to(device, non_blocking=True)
                    with torch.no_grad():
                        lbs_output = tmp_model(imgs_)
                    imgs = {"img": imgs_}
                plot_results_gt(imgs, lbs_output, f'logs/{args.dataset}/test_log.jpg')

    
    # plot the result using the best model weight
    weight_path = logger.basepath + "/model_best.pt"
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        model.eval()
        _, imgs, sketches = validation(model, optimizer, val_loader)
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

        sketches, loss = LBS_loss_fn(model, optimizer, clip_loss_fn, inputs, train_model=True)
        loss_dict.update(loss)

        loss_dict["lr"] = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if idx % args.print_every == 0:
            logger.log(
                f"[Epoch {epoch:3d} iter {idx:4d}] \t Train loss: {loss_dict['loss_total'].item():.3f}"
            )

            for name, values in loss_dict.items():
                logger.scalar_summary(name, values, steps)

    return inputs, sketches

def validation(model, optimizer, val_loader):
    val_loss = 0

    for idx, datas in enumerate(val_loader):
        if idx == 20:  # validate for 20 steps
            break

        inputs = unpack_dataloader(datas)
        with torch.no_grad():
            sketches, val_losses = LBS_loss_fn(model, optimizer, clip_loss_fn, inputs, train_model=False)
            val_loss += val_losses["loss_total"]

    val_loss /= (idx + 1)

    return val_loss, inputs, sketches


def test(test_loader, weight_path):
    checkpoint = torch.load(weight_path)
    tmp_model = SketchModel()
    tmp_model = tmp_model.to(device)
    tmp_model.load_state_dict(checkpoint)
    tmp_model.eval()
    for idx, data in enumerate(test_loader):
        img = data[0].to(device, non_blocking=True)
        img_path = data[1][0]
        with torch.no_grad():
            output = tmp_model(img)
        first_slash_index = img_path.find('/')
        result_path = os.path.join('logs', args.dataset, img_path[first_slash_index + 1:])
        directory, _ = os.path.split(result_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        tensor2image(output['sketch_black'].squeeze(), result_path)


def main():
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
    test_loader = DataLoader(test_set, shuffle=True, num_workers=8, pin_memory=True, batch_size=args.batch)

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


if __name__ == "__main__":
    main()

