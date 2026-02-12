#!/usr/bin/env python3
"""
Linear probing evaluation script for LatentMIM.
Evaluates learned representations on semantic segmentation tasks.
Supports: MADOS, m-cashew-plantation, m-SA-crop-type, and Substation datasets.
"""

import os
import sys
import json
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.functional.classification import multiclass_jaccard_index, binary_jaccard_index
import wandb

import timm.optim.optim_factory as optim_factory

from util import misc, lr_sched
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import datasets
from vits import build_vit


class LinearSegmentationHead(nn.Module):
    """Linear segmentation head for pixel-level prediction."""

    def __init__(self, embed_dim, num_classes, patch_size, use_batchnorm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.use_batchnorm = use_batchnorm

        # Linear classifier
        self.head = nn.Linear(embed_dim, num_classes)

        # Optional batch normalization
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(embed_dim)

        # Initialize weights
        nn.init.normal_(self.head.weight, mean=0, std=0.01)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x, target_size):
        """
        Args:
            x: [B, N, D] patch features from encoder
            target_size: (H, W) tuple for output size
        Returns:
            [B, num_classes, H, W] segmentation logits
        """
        B, N, D = x.shape
        H, W = target_size

        # Compute grid size
        grid_size = int(N ** 0.5)

        # Apply batchnorm if enabled
        if self.use_batchnorm:
            x = x.reshape(B * N, D)
            x = self.bn(x)
            x = x.reshape(B, N, D)

        # Apply linear classifier
        x = self.head(x)  # [B, N, num_classes]

        # Reshape to spatial grid
        x = x.permute(0, 2, 1)  # [B, num_classes, N]
        x = x.reshape(B, self.num_classes, grid_size, grid_size)

        # Upsample to target size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x


def main(args):
    if args.env.seed is not None:
        seed = args.env.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')

    ngpus_per_node = torch.cuda.device_count()
    args.env.distributed = args.env.world_size > 1 or (args.env.distributed and ngpus_per_node > 1)

    if args.env.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))
    else:
        main_worker(0, args)


def main_worker(local_rank, args):
    misc.init_distributed_mode(local_rank, args)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    job_dir = f"{args.output_dir}/{args.job_name}"
    print(f'job dir: {job_dir}')
    print("{}".format(args).replace(', ', ',\n'))

    num_tasks = misc.get_world_size()
    num_tasks_per_node = max(1, torch.cuda.device_count())
    global_rank = misc.get_rank()
    args.env.workers = args.env.workers // num_tasks_per_node

    eff_batch_size = args.batch_size * args.accum_iter * num_tasks
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    cudnn.benchmark = True

    # Create datasets based on dataset type
    print(f"Loading {args.dataset} dataset from {args.data_path}")

    if args.dataset == 'mados':
        train_dataset = datasets.mados(
            data_path=args.data_path,
            split='train',
            img_size=args.img_size,
            normalize=True,
            filter_bands=args.filter_bands
        )

        val_dataset = datasets.mados(
            data_path=args.data_path,
            split='val',
            img_size=args.img_size,
            normalize=True,
            filter_bands=args.filter_bands
        )

        test_dataset = datasets.mados(
            data_path=args.data_path,
            split='test',
            img_size=args.img_size,
            normalize=True,
            filter_bands=args.filter_bands
        )
    elif args.dataset == 'm-cashew-plantation':
        train_dataset = datasets.m_cashew_plantation(
            split='train',
            partition=args.partition,
            data_path=args.data_path,
            norm_operation=args.norm_operation,
            band_names=args.band_names,
            img_size=args.img_size
        )

        val_dataset = datasets.m_cashew_plantation(
            split='valid',
            partition=args.partition,
            data_path=args.data_path,
            norm_operation=args.norm_operation,
            band_names=args.band_names,
            img_size=args.img_size
        )

        test_dataset = datasets.m_cashew_plantation(
            split='test',
            partition=args.partition,
            data_path=args.data_path,
            norm_operation=args.norm_operation,
            band_names=args.band_names,
            img_size=args.img_size
        )
    elif args.dataset == 'm-sa-crop-type':
        train_dataset = datasets.m_sa_crop_type(
            split='train',
            partition=args.partition,
            data_path=args.data_path,
            norm_operation=args.norm_operation,
            band_names=args.band_names,
            img_size=args.img_size
        )

        val_dataset = datasets.m_sa_crop_type(
            split='valid',
            partition=args.partition,
            data_path=args.data_path,
            norm_operation=args.norm_operation,
            band_names=args.band_names,
            img_size=args.img_size
        )

        test_dataset = datasets.m_sa_crop_type(
            split='test',
            partition=args.partition,
            data_path=args.data_path,
            norm_operation=args.norm_operation,
            band_names=args.band_names,
            img_size=args.img_size
        )
    elif args.dataset == 'substation':
        train_dataset = datasets.substation(
            data_path=args.data_path,
            split='train',
            img_size=args.img_size,
            normalize=True,
        )

        val_dataset = datasets.substation(
            data_path=args.data_path,
            split='val',
            img_size=args.img_size,
            normalize=True
        )

        test_dataset = datasets.substation(
            data_path=args.data_path,
            split='test',
            img_size=args.img_size,
            normalize=True
        )
    elif args.dataset == 'tinywt':
        train_dataset = datasets.tinywt(
            data_path=args.data_path,
            split='train',
            img_size=args.img_size,
            normalize=True
        )

        val_dataset = datasets.tinywt(
            data_path=args.data_path,
            split='val',
            img_size=args.img_size,
            normalize=True
        )

        test_dataset = datasets.tinywt(
            data_path=args.data_path,
            split='test',
            img_size=args.img_size,
            normalize=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Must be one of: mados, m-cashew-plantation, m-sa-crop-type, substation, tinywt")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    if args.env.distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.env.workers,
        pin_memory=args.env.pin_mem,
        drop_last=True,
        persistent_workers=True if args.env.workers > 0 else False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.env.workers,
        pin_memory=args.env.pin_mem,
        drop_last=False,
        persistent_workers=True if args.env.workers > 0 else False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size,
        num_workers=args.env.workers,
        pin_memory=args.env.pin_mem,
        drop_last=False,
        persistent_workers=True if args.env.workers > 0 else False
    )

    # Get number of input channels from dataset
    # sample_img, _ = train_dataset[0]
    in_chans = 4 if not args.dataset == 'tinywt' else 3   # sample_img.shape[0]
    # print(f"Input channels: {in_chans}")

    # Build encoder
    encoder = build_vit(
        backbone=args.encoder,
        img_size=args.img_size,
        patch_gap=0,
        in_chans=in_chans,
        out_chans=None,
        avg_vis_mask_token=False
    )
    encoder.to(device)

    # Get encoder dimensions
    embed_dim = encoder.embed_dim
    patch_size = encoder.patchify.patch_size[0]

    # Build segmentation head
    seg_head = LinearSegmentationHead(
        embed_dim=embed_dim,
        num_classes=args.num_classes,
        patch_size=patch_size,
        use_batchnorm=args.use_batchnorm
    )
    seg_head.to(device)

    print(f"Encoder: {args.encoder}, embed_dim: {embed_dim}, patch_size: {patch_size}")
    print(f"Segmentation head: {args.num_classes} classes, use_batchnorm: {args.use_batchnorm}")

    # Wrap with DDP if distributed
    if args.env.distributed:
        encoder = DistributedDataParallel(encoder, device_ids=[args.env.gpu], find_unused_parameters=False)
        seg_head = DistributedDataParallel(seg_head, device_ids=[args.env.gpu], find_unused_parameters=False)

    encoder_without_ddp = encoder.module if hasattr(encoder, 'module') else encoder
    seg_head_without_ddp = seg_head.module if hasattr(seg_head, 'module') else seg_head

    print("Encoder = %s" % str(encoder_without_ddp))
    print("Seg head = %s" % str(seg_head_without_ddp))

    # Load pretrained encoder checkpoint
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        print(f"Loading pretrained checkpoint from {args.pretrained_checkpoint}")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')

        # Extract encoder state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Filter encoder weights
            encoder_state = {k.replace('module.encoder.', '').replace('encoder.', ''): v
                            for k, v in state_dict.items()
                            if 'encoder' in k and 'target_encoder' not in k}

            # Remove pos_embed to allow different image sizes
            encoder_state = {k: v for k, v in encoder_state.items() if 'pos_embed' not in k}

            # Remove patch embed for tinyWT which has 3 chans
            if args.dataset == 'tinywt':
                encoder_state = {k: v for k, v in encoder_state.items() if 'embed' not in k}

            msg = encoder_without_ddp.load_state_dict(encoder_state, strict=False)
            print(f"Loaded encoder with msg: {msg}")
        else:
            print("Warning: 'state_dict' not found in checkpoint")

    # Freeze encoder for linear probing
    if args.eval_mode == 'linear_probe':
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen for linear probing")
    else:
        print("Finetuning mode: encoder is trainable")

    # Setup optimizer
    if args.eval_mode == 'linear_probe':
        # Only optimize segmentation head
        no_weight_decay_list = [n for n, p in seg_head_without_ddp.named_parameters()
                               if 'bias' in n or 'norm' in n or 'bn' in n]
        param_groups = optim_factory.param_groups_weight_decay(
            seg_head_without_ddp, args.weight_decay, no_weight_decay_list=no_weight_decay_list)
    else:
        # Optimize both encoder and head (finetuning)
        # Combine parameters from both models
        combined_model = nn.ModuleDict({'encoder': encoder_without_ddp, 'seg_head': seg_head_without_ddp})
        no_weight_decay_list = [n for n, p in combined_model.named_parameters()
                               if 'bias' in n or 'norm' in n or 'bn' in n or 'embed' in n]
        param_groups = optim_factory.param_groups_weight_decay(
            combined_model, args.weight_decay, no_weight_decay_list=no_weight_decay_list)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # Setup checkpoint manager
    modules = {
        'seg_head': seg_head_without_ddp,
        'optimizer': optimizer,
        'loss_scaler': loss_scaler,
    }
    if args.eval_mode == 'finetune':
        modules['encoder'] = encoder_without_ddp

    ckpt_manager = misc.CheckpointManager(
        modules=modules,
        ckpt_dir=args.log.ckpt_dir,
        epochs=args.epochs,
        save_freq=args.log.save_freq
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_miou = 0.0
    best_val_loss = float('inf')
    if args.resume:
        start_epoch = ckpt_manager.resume()

    # Setup wandb
    if args.log.use_wandb and misc.is_main_process():
        misc.init_wandb(args, job_dir, entity=args.log.wandb_entity,
                       project=args.log.wandb_project, job_name=args.job_name)

    # Training loop
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(start_epoch, args.epochs):
        if args.env.distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_stats = train_one_epoch(
            encoder, seg_head, train_loader, optimizer, device,
            epoch, loss_scaler, args
        )

        # Validate
        val_stats = validate(encoder, seg_head, val_loader, device, args)

        # Determine if this is the best checkpoint based on configured metric
        checkpoint_metric = getattr(args, 'checkpoint_metric', 'miou')
        if checkpoint_metric == 'loss':
            is_best = val_stats['loss'] < best_val_loss
        else:  # default to miou
            is_best = val_stats['miou'] > best_val_miou

        # Update best values
        if val_stats['miou'] > best_val_miou:
            best_val_miou = val_stats['miou']
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']

        ckpt_manager.checkpoint(epoch + 1, {
            'epoch': epoch + 1,
            'best_val_miou': best_val_miou,
            'best_val_loss': best_val_loss
        })

        # Save best checkpoint separately
        if is_best and misc.is_main_process():
            ckpt_manager.checkpoint_v2('best', {
                'epoch': epoch + 1,
                'best_val_miou': best_val_miou,
                'best_val_loss': best_val_loss
            })

        # Logging
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,
            'best_val_miou': best_val_miou,
            'best_val_loss': best_val_loss
        }

        if misc.is_main_process():
            with open(os.path.join(job_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

        if args.log.use_wandb and misc.is_main_process():
            wandb.log(log_stats, step=epoch)

    # Test evaluation
    print("Running test evaluation...")

    # Load best checkpoint
    best_ckpt_path = os.path.join(args.log.ckpt_dir, 'checkpoint_best.pth')
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        ckpt_manager.load_state_dict(checkpoint)
        checkpoint_metric = getattr(args, 'checkpoint_metric', 'miou')
        if checkpoint_metric == 'loss':
            print(f"Loaded best checkpoint for testing (selected by val loss: {checkpoint.get('best_val_loss', 'N/A')})")
        else:
            print(f"Loaded best checkpoint for testing (selected by val mIoU: {checkpoint['best_val_miou']:.4f})")

    test_stats = validate(encoder, seg_head, test_loader, device, args)
    print(f"Test results - Loss: {test_stats['loss']:.4f}, mIoU: {test_stats['miou']:.4f}")

    if misc.is_main_process():
        with open(os.path.join(job_dir, 'test_results.json'), 'w') as f:
            json.dump(test_stats, f, indent=2)

        if args.log.use_wandb:
            wandb.log({'test_loss': test_stats['loss'], 'test_miou': test_stats['miou']})


def train_one_epoch(encoder, seg_head, data_loader, optimizer, device, epoch, loss_scaler, args):
    """Train for one epoch."""
    if args.eval_mode == 'linear_probe':
        encoder.eval()
    else:
        encoder.train()
    seg_head.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    for it, (images, masks) in enumerate(metric_logger.log_every(data_loader, args.log.print_freq, header)):
        # Adjust learning rate
        lr = lr_sched.adjust_learning_rate(optimizer, it / len(data_loader) + epoch, args)

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).squeeze(1)  # Dataset returns long type

        with torch.cuda.amp.autocast():
            # Get encoder features
            if args.eval_mode == 'linear_probe':
                with torch.no_grad():
                    features = encoder(images)  # [B, N, D]
            else:
                features = encoder(images)

            # Get segmentation predictions
            target_size = (masks.shape[1], masks.shape[2])
            logits = seg_head(features[:, 1:], target_size)  # [B, num_classes, H, W]

            # Compute loss
            if args.dataset == 'substation':
                loss = F.cross_entropy(
                    logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]),
                    masks.reshape(-1),
                    ignore_index=-1,
                    weight=torch.tensor([1.0, 3.0], device=logits.device)
                )
            elif args.dataset == 'tinywt':
                loss = F.cross_entropy(
                    logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]),
                    masks.reshape(-1),
                    ignore_index=-1,
                    weight=torch.tensor([1.0, 10.0], device=logits.device)
                )
            else:
                loss = F.cross_entropy(
                    logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]),
                    masks.reshape(-1),
                    ignore_index=-1
                )

            # Compute mIoU
            miou = multiclass_jaccard_index(
                logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]),
                masks.reshape(-1),
                num_classes=args.num_classes,
                ignore_index=-1
            )

            if args.dataset == "substation" or args.dataset == "tinywt":
                iou = binary_jaccard_index(
                    logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]).argmax(dim=1),
                    masks.reshape(-1),
                    ignore_index=-1
                )

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        loss /= args.accum_iter
        loss_scaler(loss, optimizer, parameters=seg_head.parameters(),
                   update_grad=(it + 1) % args.accum_iter == 0)
        if (it + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss.item() * args.accum_iter)
        metric_logger.update(miou=miou.item())
        metric_logger.update(iou=iou.item())
        metric_logger.update(lr=lr)

    # Gather stats
    metric_logger.synchronize_between_processes()
    print(f"Averaged train stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(encoder, seg_head, data_loader, device, args):
    """Validate the model."""
    encoder.eval()
    seg_head.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'

    for images, masks in metric_logger.log_every(data_loader, args.log.print_freq, header):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).squeeze(1)  # Dataset returns long type

        with torch.cuda.amp.autocast():
            # Get features
            features = encoder(images)

            # Get predictions
            target_size = (masks.shape[1], masks.shape[2])
            logits = seg_head(features[:, 1:], target_size)

            # Compute loss
            loss = F.cross_entropy(
                logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]),
                masks.reshape(-1),
                ignore_index=-1
            )

            # Compute mIoU
            miou = multiclass_jaccard_index(
                logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]),
                masks.reshape(-1),
                num_classes=args.num_classes,
                ignore_index=-1
            )

            if args.dataset == "substation" or args.dataset == "tinywt":
                iou = binary_jaccard_index(
                    logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]).argmax(dim=1),
                    masks.reshape(-1),
                    ignore_index=-1
                )

        metric_logger.update(loss=loss.item())
        metric_logger.update(miou=miou.item())
        if args.dataset == "substation" or args.dataset == "tinywt":
            metric_logger.update(iou=iou)

    # Gather stats
    metric_logger.synchronize_between_processes()
    print(f"Validation stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
