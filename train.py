#!/usr/bin/env python
""" EfficientDet Training Script

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
import argparse
import time
import yaml
from datetime import datetime

import torch
import torchvision.utils

from effdet import create_model, unwrap_bench
from data import create_loader, load_dataset_from_pickle

from timm.models import resume_checkpoint, load_checkpoint
from timm.utils import AverageMeter, CheckpointSaver, ModelEma, get_outdir

from optimizer_config import create_optimizer, create_dual_lr_optimizer
from timm.scheduler import create_scheduler
from metric.map import MeanAveragePrecision
from effdet.config import get_efficientdet_config

import wandb
import numpy as np
import logging
from torch import nn

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
PROJECT_NAME = 'BusObjectDetection'

try:
    from google.colab import drive

    google_flag = True
    print("Running on Google colab")
except:
    print("Running Locally")
    google_flag = False

torch.backends.cudnn.benchmark = True

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


def mount():
    if google_flag:
        print("Mounting Drive Folder...")
        drive.mount('/content/gdrive/')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset / Model parameters
parser.add_argument('--data',
                    default='/data/datasets/BusProject/color_data.pickle' if google_flag else '/content/gdrive/My Drive/Runners/Data/color_data.pickle',
                    help='path to dataset')
parser.add_argument('--model', default='tf_efficientdet_d1', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
add_bool_arg(parser, 'redundant-bias', default=None,
             help='override model config for redundant bias')
parser.set_defaults(redundant_bias=None)
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--no-pretrained-backbone', action='store_true', default=False,
                    help='Do not start with pretrained backbone weights, fully random.')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default='0', type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--clip-grad', type=float, default=10.0, metavar='NORM',
                    help='Clip gradient norm (default: 10.0)')

# Optimizer parameters
parser.add_argument('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "momentum"')
parser.add_argument('--opt-eps', default=1e-3, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=4e-5,
                    help='weight decay (default: 0.00004)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.08, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_fpn', type=float, default=0.004, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA amp for mixed precision training')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='map', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "map"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def disable_bn(model: nn.Module):
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.training = False


def main():
    args, args_text = _parse_args()
    mount()
    wandb.init(project=PROJECT_NAME)

    args.pretrained_backbone = not args.no_pretrained_backbone
    args.prefetcher = not args.no_prefetcher
    wandb.config.update(args)  # adds all of the arguments as config variables

    logger.info('Startubg Training with a single process on 1 GPU.')

    torch.manual_seed(args.seed)
    config = get_efficientdet_config(args.model)
    n_class = 6
    config.num_classes = n_class  # TODO:get from dataset
    model = create_model(
        config,
        bench_task='train',
        pretrained=args.pretrained,
        pretrained_backbone=args.pretrained_backbone,
        redundant_bias=args.redundant_bias,
        checkpoint_path=args.initial_checkpoint,
    )

    input_size = model.config.image_size

    logger.info('Model %s created, param count: %d' %
                (args.model, sum([m.numel() for m in model.parameters()])))

    model.cuda()

    def parameter_filter_function(name, param):
        if 'backbone' in name:
            return False
        if 'edge_weights' in name:
            return False
        if 'fpn' in name:
            return False
        return True

    def parameter_filter_b_function(name, param):
        if 'fpn' in name:
            return True
        return False

    print("Creating optimizer")
    # optimizer = create_optimizer(args, model, parameter_filter=parameter_filter_function)
    optimizer = create_dual_lr_optimizer(args, model, parameter_filter_a=parameter_filter_function,
                                         parameter_filter_b=parameter_filter_b_function, lr_b=args.lr_fpn)

    # optionally resume from a checkpoint
    resume_state = {}
    resume_epoch = None
    if args.resume:
        resume_state, resume_epoch = resume_checkpoint(unwrap_bench(model), args.resume)
    if resume_state and not args.no_resume_opt:
        if 'optimizer' in resume_state:
            # if args.local_rank == 0:
            logger.info('Restoring Optimizer state from checkpoint')
            optimizer.load_state_dict(resume_state['optimizer'])
    del resume_state

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay)
        # resume=args.resume)  # FIXME bit of a mess with bench
        if args.resume:
            load_checkpoint(unwrap_bench(model_ema), args.resume, use_ema=True)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    # if args.local_rank == 0:
    logger.info('Scheduled epochs: {}'.format(num_epochs))

    dataset_full = load_dataset_from_pickle(args.data)
    dataset_train, dataset_eval = dataset_full.split([0.8, 0.2])
    print(dataset_full.n_samples, dataset_train.n_samples, dataset_eval.n_samples)
    loader_train = create_loader(
        dataset_train,
        input_size=input_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        interpolation=args.train_interpolation,
        num_workers=args.workers,
        pin_mem=args.pin_mem,
    )

    # dataset_eval = load_dataset_from_pickle('/data/datasets/BusProject/color_data.pickle')
    loader_eval = create_loader(
        dataset_eval,
        input_size=input_size,
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        num_workers=args.workers,
        pin_mem=args.pin_mem,
    )

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None

    output_base = args.output if args.output else './output'
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model
    ])
    output_dir = get_outdir(output_base, 'train', exp_name)
    decreasing = True if eval_metric == 'loss' else False
    saver = CheckpointSaver(unwrap_bench(model), optimizer, model_ema=unwrap_bench(model_ema),
                            checkpoint_dir=output_dir, decreasing=decreasing)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()
            training_loss, class_loss, box_loss = train_epoch(
                epoch, model, loader_train, optimizer, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir, model_ema=model_ema)

            # the overhead of evaluating with coco style datasets is fairly high, so just ema or non, not both
            if model_ema is not None:
                validation_loss, (map, ap) = validate(model_ema.ema, loader_eval, args, log_suffix=' (EMA)',
                                                      n_class=n_class)
            else:
                validation_loss, (map, ap) = validate(model, loader_eval, args, n_class=n_class)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, map)

            results_dict = {'training_loss': training_loss,
                            'validation_loss': validation_loss,
                            'class_loss': class_loss,
                            'box_loss': box_loss,
                            'mAP': map}
            for api, ioui in zip(ap, np.linspace(0.5, 0.95, 10)):
                results_dict.update({f"AP{ioui}": api})

            wandb.log(results_dict)
            if saver is not None:
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=map)
            print(f"Finished Epoch {epoch} in {time.time() - start_time} with mAP:{map} ")
    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_epoch(
        epoch, model, loader, optimizer, args,
        lr_scheduler=None, saver=None, output_dir='', model_ema=None, free_bn=True):
    if args.prefetcher and args.mixup > 0 and loader.mixup_enabled:
        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            loader.mixup_enabled = False

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    class_loss_m = AverageMeter()
    box_loss_m = AverageMeter()

    model.train()
    disable_bn(model)
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        optimizer.zero_grad()
        data_time_m.update(time.time() - end)

        output = model(input, target)
        loss = output['loss']
        class_loss = output['class_loss']
        box_loss = output['box_loss']

        losses_m.update(loss.item(), input.size(0))
        class_loss_m.update(class_loss.item(), input.size(0))
        box_loss_m.update(box_loss.item(), input.size(0))

        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) / batch_time_m.val,
                    rate_avg=input.size(0) / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))
            if args.save_images and output_dir:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                unwrap_bench(model), optimizer, args, epoch, model_ema=unwrap_bench(model_ema), batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return losses_m.avg, class_loss_m.avg, box_loss_m.avg


def validate(model, loader, args, log_suffix='', n_class=1, iou_array=np.linspace(0.5, 0.95, 10)):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    evaluator = MeanAveragePrecision(n_class=n_class, iou_array=iou_array)
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # last_batch = batch_idx == last_idx

            output = model(input, target)
            loss = output['loss']

            # if evaluator is not None:
            bbox = target['bbox']
            bbox = torch.stack([bbox[:, :, 1], bbox[:, :, 0], bbox[:, :, 3], bbox[:, :, 2]],
                               dim=-1)  # Change yxyx to xyxy
            cls = torch.unsqueeze(target['cls'], dim=-1)
            max_cls = torch.sum(cls > 0, dim=1).max().item()
            target_tensor = torch.cat([bbox, cls], dim=-1)[:, :max_cls, :]
            output = output['detections']
            output = output.cpu().detach().numpy()
            output[:, :, :4] = output[:, :, :4] / target['img_scale'].reshape(-1, 1, 1).cpu().numpy()  # Normalized
            output[:, :, 2] = output[:, :, 2] + output[:, :, 0]  # Change xywh to xyxy
            output[:, :, 3] = output[:, :, 3] + output[:, :, 1]
            target_tensor = target_tensor.cpu().detach().numpy()

            evaluator.add_predictions(output, target_tensor)

            reduced_loss = loss.data

            losses_m.update(reduced_loss.item(), input.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            # if (last_batch or batch_idx % args.log_interval == 0):
            #     log_name = 'Test' + log_suffix
            #     logger.info(
            #         '{0}: [{1:>4d}/{2}]  '
            #         'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            #         'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '.format(
            #             log_name, batch_idx, last_idx, batch_time=batch_time_m, loss=losses_m))

    return losses_m.avg, evaluator.evaluate()


if __name__ == '__main__':
    main()
