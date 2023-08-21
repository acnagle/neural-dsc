import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import List

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt#; plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
import seaborn as sns#; sns.set(context='paper', style='whitegrid', font_scale=2.0, font='Times New Roman')

import fire
import horovod.torch as hvd
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data.distributed as torch_dist

from code.arch import VqvaeDiscus
from code.data import load_dataset
from code.utils import HParams, Logger, check_or_mkdir, seed_everything, get_param_count


_MARKERS = ['+', '^', 's', 'x', 'o', '*']


def pp(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs, flush=True)


def train(*,
          model_dir: str,
          seq_len: int,                         # Number of symbols jointly encoded (`L` in DISCUS).
          d_latent: int,                        # Dimension of the latent space
          bps: int = 1,                         # Bits-per-symbol in bits (`R` in DISCUS).
          enc_si: bool,                         # Whether to use side info at the encoder.
          dec_si: bool,                         # Whether to use side info at the decoder.

          # Data source parameters: Y = X + N
          p_x: float = 0.5,                     # Bernoulli paramter for data generation
          p_n: float = 0.5,                     # Bernoulli paramter for noise generation

          seed: int = 1234,
          test_run: bool = False,

          # Override default model hyperparameters
          **kwargs):
    # Argument validation
    args = {
        'model_dir': model_dir,
        'seq_len': seq_len,
        'd_latent': d_latent,
        'bps': bps,
        'enc_si': enc_si,
        'dec_si': dec_si,
        'p_x': p_x,
        'p_n': p_n,
        'seed': seed,
        'test_run': test_run
    }
    for k, v in kwargs.items():
        assert k not in args
        args[k] = v

    # Horovod setup
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # Create output directory or load checkpoint
    if hvd.rank() == 0:
        check_or_mkdir(model_dir)
        pp(f'Training in directory {model_dir}')

    # Create model
    #   - codebook_bits == bps
    seed_everything(seed)
    model = VqvaeDiscus(dim=seq_len, codebook_bits=bps, d_latent=d_latent, d_hidden=256, d_sideinfo=seq_len, enc_si=enc_si, dec_si=dec_si, **kwargs)
    model.cuda()
    hp = model.hp

    # Load data
    local_batch_size = hp.batch_size // hvd.size()
    dataset_tr = load_dataset('discus', dim=seq_len, p_x=p_x, p_n=p_n, split='train')
    sampler_tr = torch_dist.DistributedSampler(dataset_tr, num_replicas=hvd.size(), rank=hvd.rank(), seed=seed)
    loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=local_batch_size, sampler=sampler_tr, pin_memory=True)

    dataset_val = load_dataset('discus', dim=seq_len, p_x=p_x, p_n=p_n, split='val')
    sampler_val = torch_dist.DistributedSampler(dataset_val, num_replicas=hvd.size(), rank=hvd.rank(), seed=seed)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=local_batch_size*4, sampler=sampler_val, drop_last=False, pin_memory=True)

    # Create optimizer
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hp.learning_rate)
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.max_epoch * len(loader_tr), eta_min=0)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Save hparams and args
    if hvd.rank() == 0:
        logger = Logger(model_dir)
        hp.save(os.path.join(model_dir, 'hparams.json'))
        with open(os.path.join(model_dir, 'train_args.json'), 'w') as f:
            json.dump(args, f, indent=2)

    # Print some info
    params_grad = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    params_all = sum([np.prod(p.shape) for p in model.parameters()])
    pp(f'  >>> Trainable / Total params: {params_grad} / {params_all}')
    pp(f'  >>> Horovod local_size / size = {hvd.local_size()} / {hvd.size()}')
    pp(f'  >>> Per-GPU / Total batch size = {local_batch_size} / {hp.batch_size}')
    pp('Starting training!\n')

    stats = SimpleNamespace(
        loss                = [],
        loss_bler            = [],
        loss_commit         = [],
        codebook_use        = [],

        total_steps         = 0,
        steps_per_sec       = [],
        total_time          = [],
        epoch               = [],

        val_bler            = [],
        val_ber             = [],
        avg_norm_dist       = [],
        avg_norm_dist_db    = [],
    )

    # Add other task-specific params
    stats.total_rate = d_latent * bps
    stats.seq_len = seq_len
    stats.d_latent = d_latent
    stats.bps = bps

    if hvd.rank() == 0:
        logger.log_scalars({
            'task/total_rate': stats.total_rate,
            'task/seq_len': stats.seq_len,
            'task/d_latent': stats.d_latent,
            'task/bps': stats.bps,
        }, 0)

    # Full validation run
    @torch.no_grad()
    def full_eval(dataset_val, loader_val, sampler_val):
        model.eval()
        sampler_val.set_epoch(0)
        bler_list = []
        ber_list = []
        for x, y in loader_val:
            x = x.to('cuda', non_blocking=True) - 0.5
            y = y.to('cuda', non_blocking=True) - 0.5
            x_rec, _, _, _, _ = model(x, y)

            ex_err = (x_rec - x.float()).abs()
            loss_bler = (1 - (1 - ex_err).prod(axis=1)).mean()
            bler_list.append(loss_bler.cpu()[None])

            ber = (x_rec - x).abs().sum(dim=1) / seq_len
            ber_list.append(ber.cpu())

        model.train()
        bler_list = torch.cat(bler_list)
        ber_list = torch.cat(ber_list)
        assert ber_list.shape == (len(dataset_val),)
        return bler_list, ber_list

    start_time = time.time()
    total_steps = 0
    for epoch in (range(1, hp.max_epoch+1) if hp.max_epoch > 0 else itertools.count(1)):
        sampler_tr.set_epoch(epoch)
        for batch_idx, (x, y) in enumerate(loader_tr):
            total_steps += 1
            lr_scheduler.step()
            if test_run and total_steps > 10:
                break
            x = x.to('cuda', non_blocking=True) - 0.5
            y = y.to('cuda', non_blocking=True) - 0.5

            x_rec, z_q, emb, z_e, idx = model(x, y)
            ex_err = (x_rec - x.float()).abs()
            loss_bler = -torch.log(1 - ex_err + 1e-6).sum(axis=1).mean()    # block error rate loss
            loss_commit = (z_q.detach() - z_e).pow(2).mean() * hp.beta
            loss = loss_bler + loss_commit
            unique_idx = idx.unique()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Monitoring
            total_time = time.time() - start_time
            stats.loss.append(loss.item())
            stats.loss_bler.append(loss_bler.item())
            stats.loss_commit.append(loss_commit.item())
            stats.codebook_use.append(len(unique_idx.unique()))
            stats.steps_per_sec.append(total_steps / total_time)
            stats.total_steps = total_steps
            stats.total_time.append(total_time)
            stats.epoch.append(epoch)
            coeff = torch.sigmoid(model.decoder.alpha).mean()

            if total_steps % hp.print_freq == 0 or batch_idx == len(loader_tr) - 1:
                pp(f'\rep {epoch:03d} step {batch_idx+1:06d}/{len(loader_tr)} '
                   f'total_steps {total_steps:06d} ',
                   f'loss {stats.loss[-1]:.4f} ',
                   f'loss_bler {stats.loss_bler[-1]:.4f} ',
                   f'loss_commit {stats.loss_commit[-1]:.4f} ',
                   f'codebook_use {stats.codebook_use[-1]:03d} ',
                   f'time {stats.total_time[-1]:.2f} sec ',
                   f'steps/sec {stats.steps_per_sec[-1]:.2f} ',
                   f'coeff {coeff:.4f} ',
                   end='')

            if hvd.rank() == 0 and total_steps % hp.log_freq == 0:
                logger.log_scalars({
                    'train/loss': stats.loss[-1],
                    'train/loss_bler': stats.loss_bler[-1],
                    'train/loss_commit': stats.loss_commit[-1],
                    'train/codebook_use': stats.codebook_use[-1],
                    'perf/epoch': stats.epoch[-1],
                    'perf/total_time': stats.total_time[-1],
                    'perf/steps_per_sec': stats.steps_per_sec[-1],
                }, total_steps)
        pp()

        # Evaluate on validation set
        if epoch % 1 == 0:
            val_blers, val_bers = full_eval(dataset_val, loader_val, sampler_val)
            stats.val_bler.append(val_blers.mean().item())
            stats.val_ber.append(val_bers.mean().item())
            if hvd.rank() == 0:
                logger.log_scalar('eval/val_bler', stats.val_bler[-1], total_steps)
                logger.log_scalar('eval/val_ber', stats.val_ber[-1], total_steps)
                coeff = torch.sigmoid(model.decoder.alpha).mean()
                print(f'  --> [Eval] val_bler = {stats.val_bler[-1]:.4f}   val_ber = {stats.val_ber[-1]:.4f}  coeff = {coeff:.4f}')
            torch.save(vars(stats), os.path.join(model_dir, f'stats.pt'))

        if hp.ckpt_freq is not None and hvd.rank() == 0 and epoch % hp.ckpt_freq == 0:
            dump_dict = {
                'stats': vars(stats),
                'hparams': vars(hp),
                'args': args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),

                'epoch': epoch,
                'total_steps': total_steps,
            }
            torch.save(dump_dict, os.path.join(model_dir, f'ckpt_ep={epoch:03d}_step={total_steps:07d}.pt'))
            pp(f'[CHECKPOINT] Saved model at epoch {epoch}')

    pp('Training finished!')


def train_all(*,
              root_dir,
              rates: List[int],
              noise_probs: List[int],
              seq_lens: List[int],
              d_latents: List[int],
              seed: int=1234):
    if isinstance(seq_lens, int):
        seq_lens = [seq_lens]
    if isinstance(d_latents, int):
        d_latents = [d_latents]
    if isinstance(noise_probs, float):
        noise_probs = [noise_probs]
    if isinstance(rates, int):
        rates = [rates]

    assert all(type(bps) is int for bps in rates)
    assert all(type(seq_len) is int for seq_len in seq_lens)
    assert all(type(d_latent) is int for d_latent in d_latents)
    assert all(type(p_n) is float for p_n in noise_probs)

    os.makedirs(root_dir, exist_ok=True)

    num_trained = 0
    num_total = 0
    for seq_len in seq_lens:
        for d_latent in d_latents:
            for bps in rates:
                for p_n in noise_probs:
                    for mode in ['joint', 'dist', 'separate']:
                        num_total += 1
                        model_dir = os.path.join(root_dir, get_model_dir(seq_len, bps, p_n, mode))
                        stats_path = os.path.join(model_dir, 'stats.pt')
                        if os.path.exists(stats_path):
                            print(f'Skipping already trained model: {model_dir}')
                            continue
                        elif os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
                            print(f'DELETING model_dir for: {model_dir}')
                            shutil.rmtree(model_dir)

                        num_trained += 1
                        print(f'Training: {model_dir}')
                        train(
                            model_dir=model_dir,
                            seq_len=seq_len,
                            d_latent=d_latent,
                            bps=bps,
                            enc_si=(mode == 'joint'),
                            dec_si=(mode != 'separate'),
                            p_n=p_n,
                            seed=seed)

    print(f'Finished! Trained {num_trained} (total: {num_total})')


def get_model_dir(seq_len, bps, p_n, mode, run=0):
    seq_len = int(seq_len)
    bps = int(bps)
    sig_n = float(p_n)
    return f'l={seq_len}_r={bps}_probnoise={p_n}_{mode}_run={run}'


def get_best(model_dir, metric):
    stats = torch.load(os.path.join(model_dir, 'stats.pt'))

    if metric == 'dist':
        return min(stats['avg_norm_dist'])

    raise ValueError


def plot(out_path, results,
         xlabel=None, ylabel=None,
         xlim=None, ylim=None):
    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    for i, label in enumerate(results):
        if label == 'x':
            continue
        values = results[label]
        marker = _MARKERS[i]
        ax.plot(results['x'], values, label=label, marker=marker, markersize=8)
    ax.legend()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_rate_dist(*,
                   out_dir,
                   root_dir,
                   rates,
                   seq_lens,
                   sig_n,
                   modes,
                   baselines = ['wz', 'single'],
                   ):
    assert os.path.exists(root_dir) and os.path.exists(out_dir)
    if isinstance(modes, str):
        modes = [modes]
    if isinstance(baselines, str):
        baselines = [baselines]
    if isinstance(seq_lens, int):
        seq_lens = [seq_lens]

    for seq_len in seq_lens:
        for mode in modes:
            for bps in rates:
                stats_path = os.path.join(root_dir, get_model_dir(seq_len, bps, sig_n, mode), 'stats.pt')
                assert os.path.exists(stats_path), stats_path

    rates = np.array(rates, dtype=np.float32)
    for seq_len in seq_lens:
        print(f'\nL={seq_len} sig_n={sig_n} R={rates}')
        dist_wz = (sig_n ** 2) / (1 + sig_n**2) / (4 ** rates)
        print(f'WZ distortion: {dist_wz}')

        dist_single = 4.0 ** (-rates)
        print(f'Single source distortion: {dist_single}')

        results = {
            'x': rates,
            # 'Wyner-Ziv': dist_wz,
            # 'Single Source': dist_single,
        }
        if 'wz' in baselines:
            results['Optimal (with SI)'] = dist_wz
        if 'single' in baselines:
            results['Optimal (without SI)'] = dist_single

        for mode in modes:
            vals = [
                get_best(os.path.join(root_dir, get_model_dir(seq_len, bps, sig_n, mode)), 'dist')
                for bps in rates]
            label = {'joint': 'Joint', 'dist': 'Distributed', 'separate': 'Separate'}[mode]
            assert len(vals) == len(rates)
            results[label] = np.array(vals, dtype=np.float32)
            print(f'{label} VQ-VAE: {vals}')

        plot(os.path.join(out_dir, f'gaussian_rate_dist_l={seq_len:02d}_signoise={sig_n}.pdf'), results,
             ylim=(-0.01, 0.16),
            #  xlim=(1.8, 6.2),
             xlabel='Rate (bits/symbol)', ylabel='$\ell_2$ distortion')


def plot_noise_dist(*,
                    out_dir,
                    root_dir,
                    noise_stds: List[float],
                    seq_len: int,
                    bps: int = 1):
    assert os.path.exists(root_dir) and os.path.exists(out_dir)
    noise_stds = np.array(noise_stds).astype(np.float32)
    print(f'Running for {len(noise_stds)} noise stds: {noise_stds}')

    # Wyner-Ziv bound: Eqn 12 from DISCUS (2003), assuming sig_x = 1
    dist_wz = (noise_stds**2) / (1 + noise_stds**2) / (4**bps)
    print(f'WZ distortion: {dist_wz}')

    # Memoryless Gaussian source distortion: https://en.wikipedia.org/wiki/Rate%E2%80%93distortion_theory#Memoryless_(independent)_Gaussian_source_with_squared-error_distortion
    dist_single = np.ones_like(dist_wz) * 4**(-bps)
    print(f'Single source distortion: {dist_single}')

    results = {
        'Wyner-Ziv': dist_wz,
        'Single': dist_single,
    }

    # for sig_n in noise_stds:
    #     enc_si, dec_si = True, True
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         stats = train_silent(root_dir=temp_dir, seq_len=L, sig_n=sig_n, enc_si=enc_si, dec_si=dec_si)
    #     results[sig_n] = stats.distortion


def read_hparams_and_args(ckpt_path):
    ckpt_dir = os.path.dirname(ckpt_path)
    hparams = HParams.load(os.path.join(ckpt_dir, 'hparams.json'))
    with open(os.path.join(ckpt_dir, 'train_args.json'), 'r') as f:
        train_args = json.load(f)
    return hparams, train_args


@torch.no_grad()
def plot_binning(ckpt_path: str,
                 out_path: str,
                 rate: int,
                 input_range=[-10.0, 10.0],
                 num_inputs=1000):
    hp, train_args = read_hparams_and_args(ckpt_path)
    assert train_args['bps'] == rate
    assert len(input_range) == 2 and input_range[0] < input_range[1]

    model = VqvaeDiscus(**hp)
    dd = torch.load(ckpt_path)
    model.load_state_dict(dd['model_state_dict'])
    model.cuda().eval()

    x = torch.arange(input_range[0], input_range[1],
                     step=(input_range[1]-input_range[0]) / num_inputs,
                     device='cuda', dtype=torch.float32).view(-1, 1)

    #x = torch.arange(input_range[0], input_range[1],
    #                 step=(input_range[1]-input_range[0]) / num_inputs,
    #                 device='cuda', dtype=torch.float32).view(1, -1)
    dummy_side_info = torch.zeros_like(x)
    x_rec, z_q, emb, z_e, idx = model(x, c=dummy_side_info)
    assert (idx < 2 ** rate).all().item() and (idx >= 0).all().item()

    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    # ax.set_xlim(*xlim)
    # ax.set_ylim(*ylim)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # ax.set_xticks([100, 300, 500, 700])
    # ax.set_xticklabels([100, 300, 500, 700])
    # ax.plot(results['x'], values, label=label, marker=marker)
    x_np = x.cpu().numpy().flatten()
    y_np = idx.cpu().numpy().flatten().astype(np.float32)
    assert x_np.shape == y_np.shape
    ax.scatter(x_np, y_np)
    # ax.legend()
    fig.savefig(out_path + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)




if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'train_all': train_all,
        'plot_noise_dist': plot_noise_dist,
        'plot_rate_dist': plot_rate_dist,
        'plot_binning': plot_binning,
    })
