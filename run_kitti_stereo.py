import argparse
import itertools
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import List

import fire
import horovod.torch as hvd
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data.distributed as torch_dist
from torchvision.utils import save_image
from pytorch_msssim import ms_ssim as compute_ms_ssim

from code.utils import HParams, Logger, check_or_mkdir, seed_everything, get_param_count
from code.data import prepare_dataset, load_dataset
from code.arch import VqvaeKittiStereo8x, VqvaeKittiStereoBig


_ARCH_CLS = {
    'kitti_stereo_8x': VqvaeKittiStereo8x,
    'kitti_stereo_big': VqvaeKittiStereoBig,
}


def pp(*args, **kwargs):
    if hvd.rank() == 0:
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

    # Horovod setup
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # Create output directory or load checkpoint
    if ckpt is None:
        start_epoch = 1
        total_steps = 0
        if hvd.rank() == 0:
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

    local_batch_size = hp.batch_size // hvd.size()
    sampler_tr = torch_dist.DistributedSampler(dataset_tr, num_replicas=hvd.size(), rank=hvd.rank(), seed=seed)
    loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=local_batch_size, sampler=sampler_tr, **loader_kwargs)
    # sampler_val = torch_dist.DistributedSampler(dataset_val, num_replicas=hvd.size(), rank=hvd.rank())
    # loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=local_batch_size, sampler=sampler_val, **loader_kwargs)

    # Pick random samples to monitor progress.
    if hvd.rank() == 0:
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
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad])
    if hvd.rank() == 0 and ckpt is not None:
        raise NotImplementedError(f'Resuming training not supported yet!')
        # pp(f'Loading optimizer weights from ckpt...')
        # optimizer.load_state_dict(dd['optimizer_state_dict'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

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
        # bottoms = torch.cat([ex_sideinfo[:2], ex_sideinfo[:2], ex_sideinfo_wrong[:2], ex_sideinfo_random[:2],
        #                      ex_sideinfo[2:], ex_sideinfo[2:], ex_sideinfo_wrong[2:], ex_sideinfo_random[2:]], dim=0)
        # assert reconst.shape == bottoms.shape
        # reconst = torch.cat([reconst, bottoms], dim=2)
        reconst = reconst.clamp(-0.5, 0.5) + 0.5
        save_image(reconst.cpu(), os.path.join(root_dir, f'reconst_ep={epoch:03d}_step={step:07d}.png'),
                   nrow=2, padding=5, pad_value=1, range=(0, 1))

        stats.ex_mse.append((ex_input[2:] - ex_te).view(2, -1).pow(2).mean())
        stats.ex_mse_wrong.append((ex_input[2:] - ex_te_wrong).view(2, -1).pow(2).mean())
        stats.ex_mse_random.append((ex_input[2:] - ex_te_random).view(2, -1).pow(2).mean())

        # unif_idx = torch.randint(2**hp.codebook_bits, size=((4,) + hp.latent_shape), device=ex_input.device)
        # samples = model.decode_indices(unif_idx, ex_sideinfo)
        # assert samples.shape == ex_sideinfo.shape
        # samples = torch.cat([samples, ex_sideinfo], dim=2)
        # samples = samples.clamp(-0.5, 0.5) + 0.5
        # save_image(samples.cpu(), os.path.join(root_dir, f'unif_samples_ep={epoch:03d}_step={total_steps:07d}.png'),
        #            nrow=1, pad_value=1, range=(0, 1))
        model.train()

    # Save hparams and args
    if hvd.rank() == 0:
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
    pp(f'  >>> Horovod local_size / size : {hvd.local_size()} / {hvd.size()}')
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

            if hvd.rank() == 0 and total_steps % hp.log_freq == 0:
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

        if hvd.rank() == 0 and epoch % hp.sample_freq == 0:
            dump_samples(epoch, total_steps)
            logger.log_scalars({
                'examples/mse': stats.ex_mse[-1],
                'examples/mse_wrong': stats.ex_mse_wrong[-1],
                'examples/mse_random': stats.ex_mse_random[-1],
            }, total_steps)

        if hvd.rank() == 0 and epoch % hp.ckpt_freq == 0:
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
        if hvd.rank() == 0:
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
         data_root: str = None):

    ckpts = [ckpt.strip() for ckpt in ckpts.split(',')]

    assert all([os.path.basename(ckpt).startswith('ckpt_') and ckpt.endswith('.pt') for ckpt in ckpts]), \
        f'Checkpoint names must start with "ckpt_" and end with ".pt"'

    # Horovod setup
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    results = []
    for ckpt in ckpts:
        ckpt_dir = os.path.dirname(os.path.expanduser(ckpt))
        out_fn = os.path.join(ckpt_dir, f'eval_{os.path.basename(ckpt)[5:]}')
        if overwrite or not os.path.exists(out_fn):
            result_dict = eval_ckpt(ckpt, part, batch_size, data_root)
            if hvd.rank() == 0:
                torch.save(result_dict, out_fn)
        else:
            pp(f'Eval result already exists for ckpt: {ckpt}')

    pp('Evaluation finished!\n')


@torch.no_grad()
def eval_ckpt(ckpt: str = None,
              model: nn.Module = None,
              part: str = 'test',
              batch_size: int = None,
              data_root: str = None,
              hp: HParams = None):
    assert (ckpt is None) ^ (model is None)

    if ckpt:
        assert model is None
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

        if hvd.rank() == 0:
            # print(f'Loaded train args:')
            # for k, v in train_args.items():
            #     print(f'  -> {k:<20}: {v}')
            with open(os.path.join(ckpt_dir, 'eval_args.json'), 'w') as f:
                json.dump(eval_args, f, indent=2)

        # Create model
        model = _ARCH_CLS[train_args['arch']](**hp)
        dd = torch.load(os.path.expanduser(ckpt))
        model.cuda().eval()

        if hvd.rank() == 0:
            model.load_state_dict(dd['model_state_dict'])
            print(f'Loaded model weights from {ckpt}')

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

    assert batch_size % hvd.size() == 0 and len(dataset) % batch_size == 0, f'batch_size: {batch_size}, len(dataset): {len(dataset)}, hvd.size(): {hvd.size()}'
    local_batch_size = batch_size // hvd.size()
    data_sampler = torch_dist.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=local_batch_size, sampler=data_sampler, **loader_kwargs)
    shape = dataset.image_shape

    # Horovod: broadcast parameters.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    list_mse = []
    list_psnr = []
    list_ms_ssim = []

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
        mse = hvd.allgather(mse)
        psnr = hvd.allgather(psnr)
        ms_ssim = hvd.allgather(ms_ssim)
        list_mse.append(mse)
        list_psnr.append(psnr)
        list_ms_ssim.append(ms_ssim)

    all_mse = torch.cat(list_mse, dim=0).cpu()
    all_psnr = torch.cat(list_psnr, dim=0).cpu()
    all_ms_ssim = torch.cat(list_ms_ssim, dim=0).cpu()
    assert all_mse.shape == all_psnr.shape == all_ms_ssim.shape == (len(dataset),)

    # Other stats
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


def plot(results: str, labels: str, out_dir: str, prefix: str):
    results = [r.strip() for r in results.split(',')]
    labels = [l.strip() for l in labels.split(',')]
    assert len(results) == len(labels)
    for result in results:
        assert result.endswith('.pt') and os.path.isfile(result)

    assert not os.path.isfile(out_dir) and prefix
    os.makedirs(out_dir, exist_ok=True)

    for i, (result, label) in enumerate(zip(results, labels)):
        print(f'[{i:03d}] Processing label {label}: {results}')
        dd = torch.load(result)
        import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'eval': evaluate,
        'plot': plot,
    })
