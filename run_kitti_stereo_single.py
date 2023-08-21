import argparse
import glob
import itertools
import json
import os
import sys
import time
import math
from types import SimpleNamespace
from typing import List

import fire
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed as torch_dist
from torchvision.utils import save_image
from pytorch_msssim import ms_ssim as compute_ms_ssim

from code.utils import HParams, Logger, check_or_mkdir, seed_everything, get_param_count
from code.data import prepare_dataset, load_dataset
from code.arch import VqvaeKittiStereo2x, VqvaeKittiStereo4x, VqvaeKittiStereo8x, VqvaeKittiStereoBig, VqvaeKittiStereoBigOld, VqvaeKittiStereoBig2, LatentTransformer


_ARCH_CLS = {
    'kitti_stereo_2x': VqvaeKittiStereo2x,
    'kitti_stereo_4x': VqvaeKittiStereo4x,
    'kitti_stereo_8x': VqvaeKittiStereo8x,
    'kitti_stereo_big': VqvaeKittiStereoBig,
    'kitti_stereo_big_old': VqvaeKittiStereoBigOld,
    'kitti_stereo_big2': VqvaeKittiStereoBig2,
}


def pp(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def train(*,
          dataset: str = 'kitti_stereo',
          arch: str = 'kitti_stereo_8x',
          root_dir: str,
          ckpt: str = None,
          data_root: str = None,
          seed: int = 1234,
          test_run: bool = False,

          # Required model hyperparameters
          codebook_bits: int,
          ch_latent: int,
          enc_si: bool,
          dec_si: bool,

          # Override default model hyperparameters
          **kwargs):
    # Argument validation
    assert arch in _ARCH_CLS
    args = {
        'dataset': dataset,
        'arch': arch,
        'root_dir': root_dir,
        'ckpt': ckpt,
        'data_root': data_root,
        'seed': seed,

        'codebook_bits': codebook_bits,
        'ch_latent': ch_latent,
        'enc_si': enc_si,
        'dec_si': dec_si,
    }
    for k, v in kwargs.items():
        assert k not in args
        args[k] = v

    # Create output directory or load checkpoint
    if ckpt is None:
        start_epoch = 1
        total_steps = 0
        check_or_mkdir(root_dir)
        pp(f'Training in directory {root_dir}')
    else:
        raise NotImplementedError(f'Resuming training not supported yet!')

    # Load data 
    dataset_tr = load_dataset(dataset, split='train', data_root=data_root)
    dataset_val = load_dataset(dataset, split='val', data_root=data_root)

    # Create model
    seed_everything(seed)
    model = _ARCH_CLS[arch](image_shape=dataset_tr.image_shape,
                            codebook_bits=codebook_bits,
                            ch_latent=ch_latent,
                            enc_si=enc_si,
                            dec_si=dec_si, **kwargs).cuda()
    hp = model.hp

    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork'
    # to prevent # issues with Infiniband implementations that are not fork-safe
    loader_kwargs = {'num_workers': 6, 'pin_memory': True}
    if (loader_kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        loader_kwargs['multiprocessing_context'] = 'forkserver'

    local_batch_size = hp.batch_size
    sampler_tr = torch_dist.DistributedSampler(dataset_tr, num_replicas=1, rank=0, seed=seed)
    loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=local_batch_size, sampler=sampler_tr, **loader_kwargs)

    # Pick random samples to monitor progress.
    ex = [dataset_tr[i] for i in np.random.choice(len(dataset_tr), 2, replace=None)] \
        + [dataset_val[i] for i in np.random.choice(len(dataset_val), 6, replace=None)]
    ex = torch.stack(ex, dim=0).cuda().float() / 255. - 0.5
    ex_input, ex_sideinfo = ex[:4].chunk(2, dim=2)
    _, ex_sideinfo_wrong = ex[4:].chunk(2, dim=2)
    ex_sideinfo_random = torch.randn_like(ex_sideinfo).clip(-0.5, 0.5)
    assert ex_input.shape == ex_sideinfo.shape == ex_sideinfo_wrong.shape == ex_sideinfo_random.shape == \
        (4, dataset_tr.image_shape[0], dataset_tr.image_shape[1], dataset_tr.image_shape[2])

    # Create optimizer
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hp.learning_rate)

    @torch.no_grad()
    def dump_samples(epoch, step):
        model.eval()
        ex_rec, _, _, _, _ = model(ex_input, ex_sideinfo)
        ex_rec_wrong, _, _, _, _ = model(ex_input, ex_sideinfo_wrong)
        ex_rec_random, _, _, _, _ = model(ex_input, ex_sideinfo_random)
        ex_tr, ex_te = torch.split(ex_rec, 2, dim=0)
        ex_tr_wrong, ex_te_wrong = torch.split(ex_rec_wrong, 2, dim=0)
        ex_tr_random, ex_te_random = torch.split(ex_rec_random, 2, dim=0)
        reconst = torch.cat([ex_input[:2], ex_tr, ex_tr_wrong, ex_tr_random,
                             ex_input[2:], ex_te, ex_te_wrong, ex_te_random], dim=0)
        reconst = reconst.clamp(-0.5, 0.5) + 0.5
        save_image(reconst.cpu(), os.path.join(root_dir, f'reconst_ep={epoch:03d}_step={step:07d}.png'),
                   nrow=2, padding=5, pad_value=1, range=(0, 1))

        stats.ex_mse.append((ex_input[2:] - ex_te).view(2, -1).pow(2).mean())
        stats.ex_mse_wrong.append((ex_input[2:] - ex_te_wrong).view(2, -1).pow(2).mean())
        stats.ex_mse_random.append((ex_input[2:] - ex_te_random).view(2, -1).pow(2).mean())

        model.train()

    # Save hparams and args
    logger = Logger(root_dir)
    hp.save(os.path.join(root_dir, 'hparams.json'))
    with open(os.path.join(root_dir, 'train_args.json'), 'w') as f:
        json.dump(args, f, indent=2)

    # Print some info
    params_grad = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    params_all = sum([np.prod(p.shape) for p in model.parameters()])
    pp(f'  >>> Trainable / Total params  : {params_grad} / {params_all}')
    pp(f'  >>> Autoencoder params        : {model.get_autoencoder_param_count()}')
    total_rate = np.prod(hp.latent_shape).item() * hp.codebook_bits
    bpp = total_rate / hp.image_shape[1] / hp.image_shape[2]
    pp(f'  >>> BPP / Latent shape        : {bpp:.5f} / {hp.latent_shape}')
    pp(f'  >>> Horovod local_size / size : {1} / {1}')
    pp(f'  >>> Per-GPU / Total batch size: {local_batch_size} / {hp.batch_size}')
    pp('Starting training!\n')

    start_time = time.time()
    stats = SimpleNamespace(
        loss                = [],
        loss_mse            = [],
        loss_commit         = [],
        codebook_use        = [],

        steps_per_sec       = [],
        total_time          = [],
        epoch               = [],

        # For reconstructions
        ex_mse              = [],
        ex_mse_wrong        = [],
        ex_mse_random       = [],
    )

    local_steps = 0
    for epoch in (range(start_epoch, hp.max_epoch+1) if hp.max_epoch > 0 else itertools.count(start_epoch)):
        sampler_tr.set_epoch(epoch)
        for batch_idx, batch in enumerate(loader_tr):
            if test_run and local_steps > 10:
                break
            total_steps += 1
            local_steps += 1
            batch = batch.to('cuda', non_blocking=True).float() / 255. - 0.5
            assert batch.shape == (len(batch), 3, 256, 256)

            if total_steps == 1:
                save_image(batch.cpu() + 0.5, os.path.join(root_dir, f'training_sample.png'), nrow=1, padding=5, pad_value=1, range=(0, 1))

            # Split image into top (input) and bottom (side info)
            x, y = batch.chunk(2, dim=2)
            x_rec, z_q, emb, z_e, idx = model(x, y)
            loss_mse = (x_rec - x).pow(2).mean()
            loss_commit = (z_q.detach() - z_e).pow(2).mean() * hp.beta
            loss = loss_mse + loss_commit
            unique_idx = idx.unique()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Monitoring
            total_time = time.time() - start_time
            stats.loss.append(loss.item())
            stats.loss_mse.append(loss_mse.item())
            stats.loss_commit.append(loss_commit.item())
            stats.codebook_use.append(len(unique_idx.unique()))
            stats.steps_per_sec.append(local_steps / total_time)
            stats.total_time.append(total_time)
            stats.epoch.append(epoch)

            if total_steps % hp.print_freq == 0 or batch_idx == len(loader_tr) - 1:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                pp(f'\rep {epoch:03d} step {batch_idx+1:07d}/{len(loader_tr)} '
                   f'total_steps {total_steps:06d} ',
                   f'loss {stats.loss[-1]:.4f} ',
                   f'loss_mse {stats.loss_mse[-1]:.4f} ',
                   f'loss_commit {stats.loss_commit[-1]:.4f} ',
                   f'codebook_use {stats.codebook_use[-1]:03d} ',
                   f'time {stats.total_time[-1]:.2f} sec ',
                   f'steps/sec {stats.steps_per_sec[-1]:.2f} ', end='')

            if total_steps % hp.log_freq == 0:
                logger.log_scalars({
                    'train/loss': stats.loss[-1],
                    'train/loss_mse': stats.loss_mse[-1],
                    'train/loss_commit': stats.loss_commit[-1],
                    'train/codebook_use': stats.codebook_use[-1],
                    'perf/epoch': stats.epoch[-1],
                    'perf/total_time': stats.total_time[-1],
                    'perf/steps_per_sec': stats.steps_per_sec[-1],
                }, total_steps)
        pp()

        if epoch % hp.sample_freq == 0:
            dump_samples(epoch, total_steps)
            logger.log_scalars({
                'examples/mse': stats.ex_mse[-1],
                'examples/mse_wrong': stats.ex_mse_wrong[-1],
                'examples/mse_random': stats.ex_mse_random[-1],
            }, total_steps)

        if epoch % hp.ckpt_freq == 0:
            dump_dict = {
                'stats': vars(stats),
                'hparams': vars(hp),
                'args': args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),

                'epoch': epoch,
                'total_steps': total_steps,
            }
            torch.save(dump_dict, os.path.join(root_dir, f'ckpt_ep={epoch:03d}_step={total_steps:07d}.pt'))
            pp(f'[CHECKPOINT] Saved model at epoch {epoch}')

        epoch_eval_result = eval_ckpt(model=model, batch_size=128, data_root=data_root, hp=hp)
        logger.log_scalars({
            'eval/mse': epoch_eval_result['all_mse'].mean().item(),
            'eval/psnr': epoch_eval_result['all_psnr'].mean().item(),
            'eval/ms_ssim': epoch_eval_result['all_ms_ssim'].mean().item(),
            'eval/bpp': epoch_eval_result['bpp'],
            'eval/rate': epoch_eval_result['rate'],
        }, total_steps)
        pp(f'[EVAL] ep {epoch:03d} '
            f'psnr {epoch_eval_result["all_psnr"].mean().item():.4f} ',
            f'ms_ssim {epoch_eval_result["all_ms_ssim"].mean().item():.4f} ',
            f'mse {epoch_eval_result["all_mse"].mean().item():.4f} ',
            f'bpp {epoch_eval_result["bpp"]:.4f} ',
            f'rate {epoch_eval_result["rate"]} ')

        model.train()

    pp('Training finished!')


def evaluate(
         ckpts: str,
         part: str = 'test',
         overwrite: bool = False,
         batch_size: int = None,
         data_root: str = None,
         use_prior: bool = False):

    ckpts = [ckpt.strip() for ckpt in ckpts.split(',')]
    for i, ckpt in enumerate(ckpts):
        assert os.path.exists(ckpt)
        if os.path.isdir(ckpt):
            # Use the last checkpoint
            paths = []
            for path in glob.glob(os.path.join(ckpt, 'ckpt_*.pt')):
                epoch, = [int(token.split('=')[1])
                          for token in os.path.basename(path)[:-3].split('_')
                          if token.startswith('ep=')]
                paths.append((epoch, path))
            latest_ckpt = sorted(paths, key=lambda x: x[0])[-1][-1]
        else:
            latest_ckpt = ckpt
        ckpts[i] = latest_ckpt

    assert all([os.path.basename(ckpt).startswith('ckpt_') and ckpt.endswith('.pt') for ckpt in ckpts]), \
        f'Checkpoint names must start with "ckpt_" and end with ".pt"'

    print('Will process the following checkpoints:')
    for ckpt in ckpts:
        print(f'  -> {ckpt}')

    for ckpt in ckpts:
        ckpt_dir = os.path.dirname(os.path.expanduser(ckpt))
        out_fn = os.path.join(ckpt_dir, f'eval_{os.path.basename(ckpt)[5:-3]}')
        if use_prior:
            out_fn = out_fn + '_latent_prior'
        out_fn = out_fn + '.pt'
        if overwrite or not os.path.exists(out_fn):
            result_dict = eval_ckpt(ckpt=ckpt, part=part, batch_size=batch_size, data_root=data_root, use_prior=use_prior)
            torch.save(result_dict, out_fn)
            print(f'Eval result -> '
                  f'bpp: {result_dict["bpp"]} / '
                  f'psnr: {result_dict["all_psnr"].mean():.3f} / '
                  f'ssim: {result_dict["all_ms_ssim"].mean():.3f} ')
            print(f'  --> Saved to: {out_fn}')
        else:
            result_dict = torch.load(out_fn)
            pp(f'Eval result already exists for ckpt: {ckpt}')
            print(f'  -> bpp: {result_dict["bpp"]} / '
                  f'psnr: {result_dict["all_psnr"].mean():.3f} / '
                  f'ssim: {result_dict["all_ms_ssim"].mean():.3f} ')

    pp('Evaluation finished!\n')


@torch.no_grad()
def eval_ckpt(ckpt: str = None,
              model: nn.Module = None,
              part: str = 'test',
              batch_size: int = None,
              data_root: str = None,
              use_prior: bool = False,
              hp: HParams = None):
    assert (ckpt is None) ^ (model is None)

    if ckpt:
        assert model is None
        assert hp is None
        ckpt_path = os.path.expanduser(ckpt)
        ckpt_dir = os.path.dirname(ckpt_path)
        assert os.path.isfile(ckpt_path) and os.path.isdir(ckpt_dir), f'ckpt_path: {ckpt_path} ckpt_dir: {ckpt_dir}'

        # Load hparams and args from training, and save eval args
        hp = HParams.load(os.path.join(ckpt_dir, 'hparams.json'))
        with open(os.path.join(ckpt_dir, 'train_args.json'), 'r') as f:
            train_args = json.load(f)
        assert train_args['arch'] in _ARCH_CLS

        eval_args = {
            'ckpt': ckpt,
            'part': part,
            'batch_size': batch_size,
            'data_root': data_root,
        }

        with open(os.path.join(ckpt_dir, 'eval_args.json'), 'w') as f:
            json.dump(eval_args, f, indent=2)

        # Create model
        model = _ARCH_CLS[train_args['arch']](**hp)
        dd = torch.load(os.path.expanduser(ckpt))
        model.cuda().eval()

        model.load_state_dict(dd['model_state_dict'])
        print(f'Loaded model weights from {ckpt}')

        # Load in prior model
        if use_prior:
            if torch.cuda.device_count() > 1:
                prior_device = 'cuda:1'
            else:
                prior_device = 'cuda:0'

            prior_dir = os.path.join(ckpt_dir, 'learned_prior')
            
            paths = []
            for path in glob.glob(os.path.join(prior_dir, 'ckpt_*.pt')):
                epoch, = [int(token.split('=')[1])
                          for token in os.path.basename(path)[:-3].split('_')
                          if token.startswith('ep=')]
                paths.append((epoch, path))
                
            prior_ckpt = sorted(paths, key=lambda x: x[0])[-1][-1]      # use the latest checkpoint for the prior model

            prior_hp = HParams.load(os.path.join(prior_dir, 'latent_hparams.json'))

            prior_model = LatentTransformer(**vars(prior_hp))
            dd = torch.load(os.path.expanduser(prior_ckpt))
            prior_model.to(prior_device).eval()
            prior_model.load_state_dict(dd['model_state_dict'])
            print(f'Loaded prior model weights from {prior_ckpt}')
    else:
        assert model is not None
        assert hp is not None
        model.cuda().eval()

    # Load data 
    dataset = load_dataset('kitti_stereo', split=part, data_root=data_root)
    batch_size = batch_size or hp.batch_size

    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork'
    # to prevent # issues with Infiniband implementations that are not fork-safe
    loader_kwargs = {'num_workers': 6, 'pin_memory': True}
    if (loader_kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        loader_kwargs['multiprocessing_context'] = 'forkserver'

    local_batch_size = batch_size
    data_sampler = torch_dist.DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=local_batch_size, sampler=data_sampler, **loader_kwargs)
    shape = dataset.image_shape

    list_mse = []
    list_psnr = []
    list_ms_ssim = []

    if use_prior:
        total_rate = 0

    for batch_idx, batch in enumerate(data_loader):
        batch = batch.to('cuda', non_blocking=True).float() / 255. - 0.5

        # Split image into top (input) and bottom (side info)
        x, y = batch.chunk(2, dim=2)
        x_rec, z_q, emb, z_e, idx = model(x, y)
        assert x.shape == x_rec.shape == (len(x), shape[0], shape[1], shape[2])

        x = x + 0.5
        x_rec = x_rec + 0.5
        assert (x.min().item() >= 0. and x_rec.min().item() >= 0. and 
                x.max().item() <= 1. and x_rec.max().item() <= 1.)

        mse = (x_rec - x).pow(2).view(len(x), -1).sum(-1)
        psnr = -10 * torch.log10(mse / np.prod(hp.image_shape))
        ms_ssim = compute_ms_ssim(x, x_rec, size_average=False, data_range=1.0, win_size=7)
        assert mse.shape == psnr.shape == ms_ssim.shape == (len(x),)
        list_mse.append(mse)
        list_psnr.append(psnr)
        list_ms_ssim.append(ms_ssim)

        if use_prior:
            idx = idx.view(-1, hp.latent_shape[1] * hp.latent_shape[2]).to(prior_device)
            _, prob = prior_model(idx)
            prob = torch.gather(prob, dim=-1, index=idx.view(len(idx), hp.latent_shape[1] * hp.latent_shape[2], -1)).squeeze()
            total_rate += ((-prob.log2()).sum(dim=1).ceil() + 1).sum().item() / (shape[1] * shape[2])

    all_mse = torch.cat(list_mse, dim=0).cpu()
    all_psnr = torch.cat(list_psnr, dim=0).cpu()
    all_ms_ssim = torch.cat(list_ms_ssim, dim=0).cpu()
    assert all_mse.shape == all_psnr.shape == all_ms_ssim.shape == (len(dataset),)

    # Other stats
    if use_prior:
        bpp = total_rate / len(dataset)
    else:
        total_rate = np.prod(hp.latent_shape).item() * hp.codebook_bits
        bpp = total_rate / hp.image_shape[1] / hp.image_shape[2]

    result_dict = {
        'hparams': vars(hp),
        'latent_shape': hp.latent_shape,
        'codebook_bits': hp.codebook_bits,
        'rate': total_rate,
        'bpp': bpp,
        'all_mse': all_mse,
        'all_psnr': all_psnr,
        'all_ms_ssim': all_ms_ssim,
        'param_count_autoencoder': model.get_autoencoder_param_count(),
        'param_count_all': get_param_count(model),
    }

    return result_dict


def train_prior(*, vqvae_ckpt, dim: int, n_blocks: int, n_heads: int, dk: int, dv: int, batch_size: int = 12, seed=1234):
    vqvae_ckpt = os.path.expanduser(vqvae_ckpt)
    vqvae_ckpt_dir = os.path.dirname(vqvae_ckpt)
    assert os.path.isfile(vqvae_ckpt) and os.path.isdir(vqvae_ckpt_dir), f'ckpt_path: {ckpt_path} vqvae_ckpt_dir: {vqvae_ckpt_dir}'
    out_dir = os.path.join(vqvae_ckpt_dir, 'learned_prior')

    # Load hparams and args from vqvae training
    vqvae_hp = HParams.load(os.path.join(vqvae_ckpt_dir, 'hparams.json'))
    with open(os.path.join(vqvae_ckpt_dir, 'train_args.json'), 'r') as f:
        vqvae_train_args = json.load(f)
    assert vqvae_train_args['arch'] in _ARCH_CLS

    # use a second gpu to train the latent prior if one is available
    vqvae_device = 'cuda:0'
    if torch.cuda.device_count() > 1:
        prior_device = 'cuda:1'
    else:
        prior_device = 'cuda:0'

    # Create models
    vqvae_model = _ARCH_CLS[vqvae_train_args['arch']](**vqvae_hp)
    dd = torch.load(os.path.expanduser(vqvae_ckpt))
    vqvae_model.to(vqvae_device).eval()
    vqvae_model.load_state_dict(dd['model_state_dict'])
    print(f'Loaded model weights from {vqvae_ckpt}')

    prior_hp = HParams(latent_shape=vqvae_hp.latent_shape, codebook_bits=vqvae_hp.codebook_bits, dim=dim, n_blocks=n_blocks, n_heads=n_heads, dk=dk, dv=dv)
    prior_hp.add(HParams(
        epochs=100,
        bs=batch_size,
        lr=3e-4,
        clip_grad_norm=100,
        sample_freq=1,
        log_freq=20,
        print_freq=5,
        ckpt_freq=5,
        lr_warmup=2000,
        seed=seed,
    ))

    os.makedirs(out_dir, exist_ok=True)
    prior_hp.save(os.path.join(out_dir, 'latent_hparams.json'))
    logger = Logger(log_dir=out_dir)

    loader_kwargs = {'num_workers': 6, 'pin_memory': True}

    if os.path.exists(os.path.join(out_dir, 'latent_dataset_tr.pt')) and os.path.exists(os.path.join(out_dir, 'latent_dataset_test.pt')):
        dataset_tr = torch.utils.data.TensorDataset(torch.load(os.path.join(out_dir, 'latent_dataset_tr.pt')))
        sampler_tr = torch_dist.DistributedSampler(dataset_tr, num_replicas=1, rank=0, seed=seed)
        loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=prior_hp.bs, sampler=sampler_tr, **loader_kwargs)

        dataset_test = torch.utils.data.TensorDataset(torch.load(os.path.join(out_dir, 'latent_dataset_tr.pt')))
        sampler_test = torch_dist.DistributedSampler(dataset_test, num_replicas=1, rank=0, seed=seed)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=prior_hp.bs, sampler=sampler_test, **loader_kwargs)
    else:
        dataset_tr = load_dataset('kitti_stereo', split='train', data_root=None)
        sampler_tr = torch_dist.DistributedSampler(dataset_tr, num_replicas=1, rank=0, seed=seed)
        loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=1, sampler=sampler_tr, **loader_kwargs)

        dataset_test = load_dataset('kitti_stereo', split='test', data_root=None)
        sampler_test = torch_dist.DistributedSampler(dataset_test, num_replicas=1, rank=0, seed=seed)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=sampler_test, **loader_kwargs)

        latents_tr = []
        for batch_idx, batch in enumerate(loader_tr):
            batch = batch.to(vqvae_device, non_blocking=True).float() / 255. - 0.5
            assert batch.shape == (len(batch), 3, 256, 256)

            img_input, img_si = batch.chunk(2, dim=2)
            with torch.no_grad():
                _, _, _, _, x = vqvae_model(img_input, img_si)
            assert x.dtype == torch.int64 and x.shape[1:] == prior_hp.latent_shape
            x = x.view(-1, vqvae_hp.latent_shape[1] * vqvae_hp.latent_shape[2]).to(prior_device)
            latents_tr.append(x.cpu())
        latents_tr = torch.stack(latents_tr)

        latents_test = []
        for batch_idx, batch in enumerate(loader_test):
            batch = batch.to(vqvae_device, non_blocking=True).float() / 255. - 0.5
            assert batch.shape == (len(batch), 3, 256, 256)

            img_input, img_si = batch.chunk(2, dim=2)
            with torch.no_grad():
                _, _, _, _, x = vqvae_model(img_input, img_si)
            assert x.dtype == torch.int64 and x.shape[1:] == prior_hp.latent_shape
            x = x.view(-1, vqvae_hp.latent_shape[1] * vqvae_hp.latent_shape[2]).to(prior_device)
            latents_test.append(x.cpu())
        latents_test = torch.stack(latents_test)

        torch.save(latents_tr, os.path.join(out_dir, 'latent_dataset_tr.pt'))
        torch.save(latents_test, os.path.join(out_dir, 'latent_dataset_test.pt'))

        dataset_tr = torch.utils.data.TensorDataset(latents_tr)
        sampler_tr = torch_dist.DistributedSampler(dataset_tr, num_replicas=1, rank=0, seed=seed)
        loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=prior_hp.bs, sampler=sampler_tr, **loader_kwargs)

        dataset_test = torch.utils.data.TensorDataset(latents_test)
        sampler_test = torch_dist.DistributedSampler(dataset_test, num_replicas=1, rank=0, seed=seed)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=prior_hp.bs, sampler=sampler_test, **loader_kwargs)

        del vqvae_model, dd

    prior_model = LatentTransformer(**vars(prior_hp)).to(prior_device)

    optimizer = torch.optim.Adam(prior_model.parameters(), lr=prior_hp.lr)
    loss_hist = []

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=prior_hp.epochs * len(loader_tr) - prior_hp.lr_warmup, eta_min=0)

    def lr_update(step, prior_hp):
        if step <= prior_hp.lr_warmup:
            lr = lr_warmup(step)
        elif step > prior_hp.lr_warmup:
            lr_scheduler.step()
            for pg in optimizer.param_groups:
                lr = pg['lr']
        return lr

    def lr_warmup(step):
        lr = prior_hp.lr * min(step/prior_hp.lr_warmup, 1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def eval_image(data_loader, prior_model, prior_hp, vqvae_hp, prior_device):
        prior_model.eval()
        val_bpd = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                x = batch[0].squeeze(1).to(prior_device, non_blocking=True)
                logits, _ = prior_model(x)
                assert logits.shape == (len(x), vqvae_hp.latent_shape[1] * vqvae_hp.latent_shape[2], 2 ** vqvae_hp.codebook_bits)
                loss = F.cross_entropy(logits.permute(0, 2, 1), x)

                val_bpd.append(loss.item() / math.log(2))
        
        prior_model.train()
        return np.mean(val_bpd[-20:])

    stats = SimpleNamespace(
        prior_hp        = vars(prior_hp),

        loss            = [],
        bpd             = [],
        lr              = [],
        grad_norm       = [],
        total_time      = [],
        valid_bpd       = [],
    )
    print('Starting training...')
    step = 0
    start_time = time.time()
    epoch_gen = range(1, prior_hp.epochs+1) if prior_hp.epochs > 0 else itertools.count(1)
    prior_model.train()
    for epoch in epoch_gen:
        sampler_tr.set_epoch(epoch)
        for batch_idx, batch in enumerate(loader_tr):
            step += 1
            lr = lr_update(step, prior_hp)
            x = batch[0].squeeze(1).to(prior_device, non_blocking=True)
            optimizer.zero_grad()
            logits, _ = prior_model(x)
            assert logits.shape == (len(x), vqvae_hp.latent_shape[1] * vqvae_hp.latent_shape[2], 2 ** vqvae_hp.codebook_bits)
            loss = F.cross_entropy(logits.permute(0, 2, 1), x)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(prior_model.parameters(), prior_hp.clip_grad_norm)
            optimizer.step()

            # Log training stats
            stats.loss.append(loss.item())
            stats.bpd.append(loss.item() / math.log(2))
            stats.lr.append(lr)
            stats.grad_norm.append(grad_norm)
            stats.total_time.append(time.time()-start_time)

            if step % prior_hp.log_freq == 0:
                logger.log_scalars({
                    'loss/loss': stats.loss[-1],
                    'loss/bpd': stats.bpd[-1],
                    'optim/lr': stats.lr[-1],
                    'optim/grad_norm': stats.grad_norm[-1],
                    'progress/epoch': epoch,
                    'progress/step': step,
                    'progress/total_time': stats.total_time[-1],
                }, step)
                torch.save(stats, os.path.join(out_dir, 'stats.pt'))

            if step % prior_hp.print_freq == 0:
                print('\r'
                      f'ep {epoch} '
                      f'step {step} '
                      f'loss {np.mean(stats.loss[-20:]):.5f} '
                      f'bpd {np.mean(stats.bpd[-20:]):.5f} '
                      f'lr {lr:.3e} '
                      f'grad_norm {grad_norm:.2f} '
                      f'time {time.time()-start_time:.2f}',
                      end='', flush=True)


        # Eval on validation set
        val_bpd = eval_image(loader_test, prior_model, prior_hp, vqvae_hp, prior_device)
        print(f'\nep {epoch} step {step} val_bpd {val_bpd:.4f}')
        stats.valid_bpd.append(val_bpd)
        logger.log_scalar('eval/val_bpd', val_bpd, step)

        dump_dict = {
            'model_state_dict': prior_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': vars(stats),
            'epoch': epoch,
            'step': step,
            'hparams': vars(prior_hp),
            'elapsed_time': time.time() - start_time
        }

        if epoch % prior_hp.ckpt_freq == 0:
            torch.save(dump_dict, os.path.join(out_dir, f'ckpt_ep={epoch:03d}_step={step:07d}.pt'))
            pp(f'[CHECKPOINT] Saved model at epoch {epoch}')


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'train_prior': train_prior,
        'eval': evaluate,
    })
