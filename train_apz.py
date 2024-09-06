# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import gc

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import math

from download import find_model
from apz_models import apz_DiT_models
from models.diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# Prog
from prog.progressive import progressive_schedule, make_divisible
from prog.metrics import AverageMeter, accuracy, SmoothMeter
from prog.helpers import *
from scipy.optimize import curve_fit
import random
import torch.nn.functional as F

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX # 模糊滤波
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC # 双三次插值
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    args.local_rank = rank
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    def create_model(latent_size):
        model = apz_DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        )
        # model = DiT_models[args.model]()
        # load pre-trained model
        ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        state_dict = find_model(ckpt_path)
        # model.load_state_dict(state_dict)
        model.load_pretrained(state_dict)
        return model

    model = create_model(latent_size)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    total_time = 0
    stage = 0

    args.num_per_epoch_steps = len(loader) // args.gradient_accumulation_steps
    args.epochs = math.ceil(args.max_train_steps / args.num_per_epoch_steps)

    logger.info(f"Training for {args.epochs} epochs...")
    logger.info(f"Global batch size: {args.global_batch_size}, Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"per GPU data loader length: {len(loader)}")
    logger.info(f"num per epoch steps: {args.num_per_epoch_steps}")

    # setup progressive schedule
    r_max = args.image_size
    grow_epochs, r_list, l_list = progressive_schedule(args,
                                              r_max=r_max)

    logger.info(f'Progressive training settings:\n\t'
                f'stage number  :\t{args.num_stages}\n\t'
                f'grow epochs   :\t{grow_epochs}\n\t'
                f'resolution    :\t{r_list}\n\t'
                f'layer number  :\t{l_list}\n\t'
                )
    current_r = r_list[0]
    current_l = l_list[0]
    prev_r, prev_l = current_r, current_l
    # prepare zero-shot nas
    loader = DataLoader(
        dataset,
        batch_size=int(16 // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    for epoch in range(args.epochs):

        if epoch in grow_epochs:
            stage = grow_epochs.index(epoch)

            # current_r, current_l = r_list[stage], l_list[stage]

            if args.auto_grow and stage < len(grow_epochs) - 1:

                search_r_list, search_l_list = no_repeats(r_list), no_repeats(l_list)
                if stage > 0:
                    r_s, l_s = search_r_list.index(current_r), search_l_list.index(current_l)
                    prev_r, prev_l = current_r, current_l
                    # if l_s < len(search_l_list) - 1:
                    #     l_s += 1

                    r_e, l_e = min(r_s + 2, len(search_r_list)), min(l_s + 2,
                                                                     len(search_l_list))  # search for max of 3 candidates
                    search_r_list, search_l_list = search_r_list[r_s:r_e], search_l_list[l_s:l_e]
                else:

                    search_r_list, search_l_list = [search_r_list[0], search_r_list[len(search_r_list) // 2 - 1],
                                                    ], \
                        [search_l_list[0], search_l_list[len(search_l_list) // 2 - 1], ]
                max_r, max_l = search_r_list[-1], search_l_list[-1]

                if current_r != max_r or current_l != max_l:
                    if args.local_rank == 0:
                        logger.info(f'auto grow started, grow range: {current_r} -> {max_r}')

                    current_r, current_l = \
                        auto_grow(args, r_list=search_r_list, l_list=search_l_list,
                                  epoch=epoch, stage=stage, model=model, ema=ema,
                                  train_dataloader=loader, device=device, vae=vae,
                                  diffusion=diffusion, opt=opt, logger=logger, train_steps=train_steps
                                  )
            else:
                current_r, current_l = r_list[stage], l_list[stage]


            torch.cuda.empty_cache()
            set_model_config(model, current_l)
            # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            logger.info(f"current layer: {current_l}")

            # Setup data:
            _transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, current_r)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            _dataset = ImageFolder(args.data_path, transform=_transform)
            _sampler = DistributedSampler(
                _dataset,
                num_replicas=dist.get_world_size(),
                rank=rank,
                shuffle=True,
                seed=args.global_seed
            )
            _loader = DataLoader(
                _dataset,
                batch_size=int(args.global_batch_size // dist.get_world_size()),
                shuffle=False,
                sampler=_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True
            )

            logger.info(f"current resolution: {current_r}")


        # gc.collect()
        _sampler.set_epoch(epoch)
        # logger.info(f"Beginning epoch {epoch}...")
        for i, (x, y) in enumerate(_loader):

            if i >= args.num_per_epoch_steps * args.gradient_accumulation_steps:  # 不够完整的梯度累积就跳出
                break

            # print(x.shape, y.shape)
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            s = torch.full((x.shape[0],), stage, dtype=torch.long, device=device)
            model_kwargs = dict(y=y, s=s)

            _start_time = time()

            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
            opt.zero_grad()
            loss.backward()

            _end_time = time()
            total_time += _end_time - _start_time

            if (i + 1) % args.gradient_accumulation_steps == 0:
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                opt.step()
                update_ema(ema, model.module)

                # Log loss values:
                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Train time: {total_time / 3600:.2f}")
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save DiT checkpoint:
                if train_steps % args.ckpt_every == 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()

            if train_steps >= args.max_train_steps:
                break

    if rank == 0:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info(f"Train total_time: {total_time / 3600}")
    logger.info("Done!")
    cleanup()


def auto_grow(args, r_list, l_list, epoch, stage,
              model, ema, train_dataloader, device,
              vae, diffusion, opt,logger, train_steps):

    cfg_strs = []
    for r in r_list:
        for l in l_list:
            cfg_strs.append(f'r{r}_l{l}')
    if args.local_rank == 0:
        logger.info(f'r list: {list(r_list)}; l list: {list(l_list)} \n cfg list:{cfg_strs}')



    search_metrics = train_one_epoch_super(
        args,
        epoch,
        stage,
        model,
        ema,
        train_dataloader,
        device,
        vae,
        diffusion,
        opt,
        logger,
        train_steps,
        l_list=l_list,
        r_list=r_list,
        cfg_strs=cfg_strs,
    )

    loss_d = {}
    taylor0_loss_d = {}
    time_d = {}
    zico_d = {}
    ntk_d = {}
    for cfg in cfg_strs:
        time_list = [search_metrics[i][cfg]['time'] for i in range(len(search_metrics))]
        ntk_list = [search_metrics[i][cfg]['ntk'] for i in range(len(search_metrics))]
        zico_list = [search_metrics[i][cfg]['zico'] for i in range(len(search_metrics))]
        # taylor0_time = sum(time_list) / len(time_list)
        taylor0_ntk = sum(ntk_list) / len(ntk_list)
        taylor0_zico = sum(zico_list) / len(zico_list)
        time_d[cfg] = time_list[-1]
        ntk_d[cfg] = taylor0_ntk
        zico_d[cfg] = taylor0_zico

    def get_rankings(data_dict, reverse=True):
        # 根据值对键进行排序，并处理相同值的情况
        sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=reverse)
        rankings = {}
        current_rank = 1  # 当前排名
        previous_value = None  # 前一个值
        for i, (key, value) in enumerate(sorted_items):
            if previous_value == None:
                previous_value = value
            elif abs(value - previous_value) > 1e-6:  # 如果值改变了，更新当前的排名为列表索引+1
                current_rank = i + 1
                previous_value = value
            rankings[key] = current_rank  # 分配排名
        return rankings

    time_sorted = get_rankings(time_d, reverse=False)
    ntk_sorted = get_rankings(ntk_d, reverse=True)
    zico_sorted = get_rankings(zico_d, reverse=True)

    # 计算综合排名
    all_sorted_d = defaultdict(int)
    # for rankings in [time_sorted, ntk_sorted, zico_sorted]:
    #     for key, rank in rankings.items():
    #         all_sorted_d[key] += rank

    for key, rank in time_sorted.items():
        all_sorted_d[key] += rank
    for key, rank in ntk_sorted.items():
        all_sorted_d[key] += rank * 0.5
    for key, rank in zico_sorted.items():
        all_sorted_d[key] += rank * 0.5


    all_sorted = sorted(all_sorted_d, key=all_sorted_d.get, reverse=False)
    if args.local_rank == 0:
        logger.info('\nTime: ' + '; '.join(['{}: {:>7.4f}  '.format(k, time_d[k]) for k in time_sorted]) +
                    '\nNTK: ' + '; '.join(['{}: {:>7.4f}  '.format(k, ntk_d[k]) for k in ntk_sorted]) +
                    '\nZico: ' + '; '.join(['{}: {:>7.4f}  '.format(k, zico_d[k]) for k in zico_sorted]) +
                    '\nAll: ' + '; '.join(['{}: {:>7.4f}  '.format(k, all_sorted_d[k]) for k in all_sorted])
                    )

    torch.cuda.synchronize()


    best_r = int(all_sorted[0].split('_')[0].lstrip('r'))
    best_l = int(all_sorted[0].split('_')[1].lstrip('l'))

    return best_r, best_l

def train_one_epoch_super(args,
                          epoch,
                          stage,
                          model,
                          ema,
                          train_dataloader,
                          device,
                          vae,
                          diffusion,
                          opt,
                          logger,
                          train_steps,
                          l_list=None,
                          r_list=None,
                          cfg_strs=None,
                          ):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    search_metrics = []
    search_m = {}
    losses_m = []
    batch_time_m = []
    NTK_dict_m = []
    Zico_dict_m = []
    if l_list is None:
        l_list = [1]
    if r_list is None:
        r_list = [256]
    for j in range(len(r_list)):
        losses_m.append([SmoothMeter() for i in range(len(l_list))])
        batch_time_m.append([AverageMeter() for i in range(len(l_list))])
        NTK_dict_m.append([[] for j in range(len(l_list))])
        Zico_dict_m.append([[] for j in range(len(l_list))])
    last_idx = args.num_per_epoch_steps * args.gradient_accumulation_steps

    # original_state_dict = {name: param.clone().detach().to('cpu') for name, param in unet.state_dict().items()}
    search_time = 0
    total_search_epochs = args.search_epochs
    for search_epoch in range(epoch, epoch + total_search_epochs):

        train_loss = 0.0
        for r in r_list:
            for l in l_list:
                set_model_config(model, l)
                r_idx = r_list.index(r)
                l_idx = l_list.index(l)
                grad_dict = defaultdict(list)

                for i, (x, y) in enumerate(train_dataloader):
                    if i < 2:
                        x = x.to(device)
                        y = y.to(device)
                        end = time()
                        x = F.interpolate(x, size=(r, r), mode='bilinear',
                                         align_corners=False)

                        with torch.no_grad():
                            # Map input images to latent space + normalize latents:
                            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                        s = torch.full((x.shape[0],), stage, dtype=torch.long, device=device)
                        model_kwargs = dict(y=y, s=s)
                        data_end = time()
                        data_time_m.update(data_end - end)
                        start = time()
                        # ntk
                        ntk = compute_NTK_score(model, x, t, model_kwargs, diffusion, opt)
                        NTK_dict_m[r_idx][l_idx].append(ntk)
                        search_time += time() - start
                        opt.zero_grad()

                    elif i < 4:
                        x = x.to(device)
                        y = y.to(device)
                        end = time()
                        x = F.interpolate(x, size=(r, r), mode='bilinear',
                                         align_corners=False)

                        with torch.no_grad():
                            # Map input images to latent space + normalize latents:
                            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                        s = torch.full((x.shape[0],), stage, dtype=torch.long, device=device)
                        model_kwargs = dict(y=y, s=s)
                        data_end = time()
                        data_time_m.update(data_end - end)
                        start = time()
                        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                        loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
                        loss.backward()

                        # zico
                        grad_dict = getgrad(model, grad_dict)
                        opt.zero_grad()
                        search_time += time() - start
                    elif i < 10:
                        x = x.to(device)
                        y = y.to(device)
                        end = time()
                        x = F.interpolate(x, size=(r, r), mode='bilinear',
                                           align_corners=False)
                        with torch.no_grad():
                            # Map input images to latent space + normalize latents:
                            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                        s = torch.full((x.shape[0],), stage, dtype=torch.long, device=device)
                        model_kwargs = dict(y=y, s=s)
                        data_end = time()
                        data_time_m.update(data_end - end)
                        start = time()
                        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

                        loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
                        opt.zero_grad()
                        loss.backward()

                        r_idx = r_list.index(r)
                        l_idx = l_list.index(l)
                        batch_time_m[r_idx][l_idx].update(time() - start)


                zico = caculate_zico(grad_dict)
                Zico_dict_m[r_idx][l_idx].append(zico)
                if args.local_rank == 0:
                    print("ok")
                opt.zero_grad()
                torch.cuda.empty_cache()
                dist.barrier()


        search_m = {}
        for cfg in cfg_strs:
            r_idx = r_list.index(int(cfg.split('_')[0].lstrip('r')))
            l_idx = l_list.index(int(cfg.split('_')[1].lstrip('l')))
            search_m[cfg] = OrderedDict(
                [('time', batch_time_m[r_idx][l_idx].avg),
                 ('ntk', np.mean(NTK_dict_m[r_idx][l_idx])),
                 ('zico', np.mean(Zico_dict_m[r_idx][l_idx]))]
            )
        search_metrics.append(deepcopy(search_m))

    logger.info(f"search time: {search_time}")

    return search_metrics


def sample_configs(l_list, r_list, mode='random'):
    if mode == 'random':
        config = {'min_layer_num': l_list[0], 'max_layer_num': l_list[-1], 'layer_num': random.choice(l_list),
                  'input_size': random.choice(r_list)}

    elif mode == 'smallest':
        config = {'min_layer_num': l_list[0], 'max_layer_num': l_list[-1], 'layer_num': l_list[0],
                  'input_size': r_list[0]}
    elif mode == 'largest':
        config = {'min_layer_num': l_list[0], 'max_layer_num': l_list[-1], 'layer_num': l_list[-1],
                  'input_size': r_list[-1]}

    else:
        raise NotImplementedError

    return config, l_list.index(config['layer_num']), r_list.index(config['input_size'])

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(apz_DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=20_000)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--max_train_steps", type=int, default=240_000)

    # Prog
    parser.add_argument('--auto_grow', default=False, action='store_true', help='auto grow')
    parser.add_argument('--search_epochs', type=int, default=2, help='epochs for each auto grow search')

    parser.add_argument('--r_scale', type=float, default=0.25, help='smallest scale of resolution')
    parser.add_argument('--l_scale', type=float, default=0.25, help='smallest scale of layer num')
    parser.add_argument('--num_stages', type=int, default=4, help='progressive stages')
    args = parser.parse_args()
    main(args)
