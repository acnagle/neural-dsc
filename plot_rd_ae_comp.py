import argparse
import glob
import os
from typing import Dict

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
import seaborn as sns; sns.set(context='paper', style='white', font_scale=2.0, font='Times New Roman')

import numpy as np
import torch


_MARKERS = ['o', '^', 's', 'x', 'm']


FIGSIZE = (8, 4)


def plot_rate_psnr(*, out_file: str, results: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
    xticks = set()

    for i, (label, files) in enumerate(results.items()):
        rate_psnr = []
        for fn in files:
            res = torch.load(fn)
            if res['bpp'] >= 0.7:
                continue
            rate_psnr.append((res['bpp'], res['psnr'].mean().item()))
            xticks.add(res['bpp'])
        rate_psnr = sorted(rate_psnr, key=lambda x: x[0])
        x, y = list(zip(*rate_psnr))
        ax.plot(x, y, label=label, markersize=5, linewidth=1, marker=_MARKERS[i])
        print(f'Plot {label} PSNR values: {y}')

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('BPP')
    ax.set_ylabel('PSNR')
    # ax.set_ylim(41, 47.5)
    # ax.set_xscale('log')
    # ax.set_xticks(list(xticks))
    # ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def plot_rate_ms_ssim(*, out_file: str, results: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
    xticks = set()

    for i, (label, files) in enumerate(results.items()):
        rate_ms_ssim = []
        for fn in files:
            res = torch.load(fn)
            if res['bpp'] >= 0.7:
                continue
            rate_ms_ssim.append((res['bpp'], res['ms_ssim'].mean().item()))
            xticks.add(res['bpp'])
        rate_ms_ssim = sorted(rate_ms_ssim, key=lambda x: x[0])
        ax.plot(*zip(*rate_ms_ssim), label=label, markersize=5, linewidth=1, marker=_MARKERS[i])

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('BPP')
    ax.set_ylabel('MS-SSIM')
    # ax.set_xscale('log')
    # ax.set_xticks(list(xticks))
    # ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)



plot_rate_psnr(
    out_file='paper/ae_comp_celebahq256_top_rate_psnr.pdf',
    results={
        'VQ-VAE':       glob.glob('checkpoints/celebahq256_cvqvae_top_dist_[12345678]bit/eval_*.pt'),
        # 'AE ($D=32$)':  glob.glob('checkpoints/celebahq256_ae_top_dist/eval_*.pt'),
        'AE ($D=4$)':   glob.glob('checkpoints/celebahq256_ae_top_dist_emb4/eval_*.pt'),
        'AE ($D=1$)':   glob.glob('checkpoints/celebahq256_ae_top_dist_emb1/eval_*.pt'),
    }
)

plot_rate_ms_ssim(
    out_file='paper/ae_comp_celebahq256_top_rate_ms_ssim.pdf',
    results={
        'VQ-VAE':       glob.glob('checkpoints/celebahq256_cvqvae_top_dist_[12345678]bit/eval_*.pt'),
        # 'AE ($D=32$)':  glob.glob('checkpoints/celebahq256_ae_top_dist/eval_*.pt'),
        'AE ($D=4$)':   glob.glob('checkpoints/celebahq256_ae_top_dist_emb4/eval_*.pt'),
        'AE ($D=1$)':   glob.glob('checkpoints/celebahq256_ae_top_dist_emb1/eval_*.pt'),
    }
)

