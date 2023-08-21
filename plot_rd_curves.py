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


def load_csv(fpath: str):
    with open(fpath, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    headers = lines[0].split(',')
    out = {k: [] for k in headers}
    for row in lines[1:]:
        values = row.split(',')
        assert len(values) == len(headers)
        for i in range(len(values)):
            if '.' in values[i]:
                values[i] = float(values[i])
            else:
                values[i] = int(values[i])
        for k, v in zip(headers, values):
            out[k].append(v)
    return out


def plot_rate_psnr(*, out_file: str, results: Dict, others: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1)
    xticks = set()
    plot_count = 0

    for label, files in results.items():
        rate_psnr = []
        for fn in files:
            res = torch.load(fn)
            if res['rate'] > 2048:
                continue
            rate_psnr.append((res['rate'], res['psnr'].mean().item()))
            xticks.add(res['rate'])
        rate_psnr = sorted(rate_psnr, key=lambda x: x[0])
        x, y = list(zip(*rate_psnr))
        ax.plot(x, y, label=label, markersize=5, linewidth=1, marker=_MARKERS[plot_count])
        print(f'Plot {label} PSNR values: {y}')
        plot_count += 1

    for label, files in others.items():
        rate_psnr = []
        for fn in files:
            res = load_csv(fn)
            rate_psnr.append((np.mean(res['rate']).item(), np.mean(res['psnr'])))
        rate_psnr = sorted(rate_psnr, key=lambda x: x[0])
        x, y = list(zip(*rate_psnr))
        ax.plot(x, y, label=label, markersize=5, linewidth=1, marker=_MARKERS[plot_count])
        print(f'Plot {label} PSNR values: {y}')
        plot_count += 1

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Rate (bits)')
    ax.set_ylabel('PSNR')
    # ax.set_ylim(41, 47.5)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def plot_rate_mse(*, out_file: str, results: Dict, others: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1)
    xticks = set()
    plot_count = 0

    for i, (label, files) in enumerate(results.items()):
        rate_mse = []
        for fn in files:
            res = torch.load(fn)
            if res['rate'] > 2048:
                continue
            rate_mse.append((res['rate'], res['mse'].mean().item()))
            xticks.add(res['rate'])
        rate_mse = sorted(rate_mse, key=lambda x: x[0])
        ax.plot(*zip(*rate_mse), label=label, markersize=5, linewidth=1, marker=_MARKERS[plot_count])
        plot_count += 1

    for label, files in others.items():
        rate_mse = []
        for fn in files:
            res = load_csv(fn)
            rate_mse.append((np.mean(res['rate']).item(), np.mean(res['mse'])))
        rate_mse = sorted(rate_mse, key=lambda x: x[0])
        x, y = list(zip(*rate_mse))
        ax.plot(x, y, label=label, markersize=5, linewidth=1, marker=_MARKERS[plot_count])
        plot_count += 1

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Rate (bits)')
    ax.set_ylabel('MSE')
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def plot_rate_ms_ssim(*, out_file: str, results: Dict, others: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1)
    xticks = set()
    plot_count = 0

    for i, (label, files) in enumerate(results.items()):
        rate_ms_ssim = []
        for fn in files:
            res = torch.load(fn)
            rate_ms_ssim.append((res['rate'], res['ms_ssim'].mean().item()))
            xticks.add(res['rate'])
        rate_ms_ssim = sorted(rate_ms_ssim, key=lambda x: x[0])
        ax.plot(*zip(*rate_ms_ssim), label=label, markersize=5, linewidth=1, marker=_MARKERS[plot_count])

    for label, files in others.items():
        rate_ms_ssim = []
        for fn in files:
            res = load_csv(fn)
            rate_ms_ssim.append((np.mean(res['rate']).item(), np.mean(res['ssim'])))
        rate_ms_ssim = sorted(rate_ms_ssim, key=lambda x: x[0])
        x, y = list(zip(*rate_ms_ssim))
        ax.plot(x, y, label=label, markersize=5, linewidth=1, marker=_MARKERS[plot_count])
        plot_count += 1

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Rate (bits)')
    ax.set_ylabel('MS-SSIM')
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def plot_bpp_psnr(*, out_file: str, results: Dict, others: Dict = {}):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1)
    # xticks = set()
    plot_count = 0

    for label, files in results.items():
        bpp_psnr = []
        for fn in files:
            res = torch.load(fn)
            if res['rate'] > 2048:
                continue
            bpp_psnr.append((res['bpp'], res['psnr'].mean().item()))
            # xticks.add(res['bpp'])
        bpp_psnr = sorted(bpp_psnr, key=lambda x: x[0])
        x, y = list(zip(*bpp_psnr))
        ax.plot(x, y, label=label, markersize=5, linewidth=1, marker=_MARKERS[plot_count])
        print(f'Plot {label} PSNR values: {y}')
        plot_count += 1

    for label, files in others.items():
        bpp_psnr = []
        for fn in files:
            res = load_csv(fn)
            bpp_psnr.append((np.mean(res['bpp']).item(), np.mean(res['psnr'])))
        bpp_psnr = sorted(bpp_psnr, key=lambda x: x[0])
        x, y = list(zip(*bpp_psnr))
        ax.plot(x, y, label=label, markersize=5, linewidth=1, marker=_MARKERS[plot_count])
        print(f'Plot {label} PSNR values: {y}')
        plot_count += 1

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Rate (bits per pixel)')
    ax.set_ylabel('Distortion (PSNR)')
    # ax.set_ylim(41, 47.5)
    # ax.set_xticks(list(xticks))
    # ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)



def plot_bpp_ms_ssim(*, out_file: str, results: Dict, others: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1)
    # xticks = set()
    plot_count = 0

    for i, (label, files) in enumerate(results.items()):
        bpp_ms_ssim = []
        for fn in files:
            res = torch.load(fn)
            bpp_ms_ssim.append((res['bpp'], res['ms_ssim'].mean().item()))
            # xticks.add(res['bpp'])
        rate_mse = sorted(bpp_ms_ssim, key=lambda x: x[0])
        ax.plot(*zip(*bpp_ms_ssim), label=label, markersize=4, linewidth=1, marker=_MARKERS[plot_count])
        plot_count += 1

    for label, files in others.items():
        bpp_ms_ssim = []
        for fn in files:
            res = load_csv(fn)
            bpp_ms_ssim.append((np.mean(res['bpp']).item(), np.mean(res['ssim'])))
        bpp_ms_ssim = sorted(bpp_ms_ssim, key=lambda x: x[0])
        x, y = list(zip(*bpp_ms_ssim))
        ax.plot(x, y, label=label, markersize=5, linewidth=1, marker=_MARKERS[plot_count])
        plot_count += 1

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('BPP')
    ax.set_ylabel('MS-SSIM')
    # ax.set_xticks(list(xticks))
    # ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


NDIC_RESULT_FILES = [
    # 'ndic_exp/eval_results/bmshj18.celebahq.ms_ssim.lambda_0.00007/eval_result.csv',
    # 'ndic_exp/eval_results/bmshj18.celebahq.ms_ssim.lambda_0.0001/eval_result.csv',
    # 'ndic_exp/eval_results/bmshj18.celebahq.ms_ssim.lambda_0.0005/eval_result.csv',
    # 'ndic_exp/eval_results/bmshj18.celebahq.ms_ssim.lambda_0.001/eval_result.csv',
    # 'ndic_exp/eval_results/bmshj18.celebahq.mse.lambda_0.00002/eval_result.csv',
    'ndic_exp/eval_results/bmshj18.celebahq.mse.lambda_0.00005/eval_result.csv',
    'ndic_exp/eval_results/bmshj18.celebahq.mse.lambda_0.0001/eval_result.csv',
    'ndic_exp/eval_results/bmshj18.celebahq.mse.lambda_0.0005/eval_result.csv',
    'ndic_exp/eval_results/bmshj18.celebahq.mse.lambda_0.001/eval_result.csv',
    # 'ndic_exp/eval_results/bmshj18.celebahq.mse.lambda_0.005/eval_result.csv',
]


plot_rate_psnr(
    out_file='paper/celebahq256_top_rate_psnr.pdf',
    results={
        'Joint'         : glob.glob('checkpoints/celebahq256_cvqvae_top_joint_[1234]bit/eval_*.pt'),
        'Distributed'   : glob.glob('checkpoints/celebahq256_cvqvae_top_dist_[1234]bit/eval_*.pt'),
        'Separate'      : glob.glob('checkpoints/celebahq256_cvqvae_top_separate_[1234]bit/eval_*.pt'),
    },
    others={
        # 'NDIC'          : glob.glob('ndic_exp/eval_results/bmshj18.celebahq.ms_ssim.lambda_*/eval_result.csv'),
        'NDIC'          : NDIC_RESULT_FILES,
    }
)

# plot_rate_mse(
#     out_file='paper/celebahq256_top_rate_mse.pdf',
#     results={
#         'Joint'         : glob.glob('checkpoints/celebahq256_vqvae_joint_[1234]bit/eval_*.pt'),
#         'Distributed'   : glob.glob('checkpoints/celebahq256_vqvae_dist_[1234]bit/eval_*.pt'),
#         'Separate'      : glob.glob('checkpoints/celebahq256_vqvae_separate_[1234]bit/eval_*.pt'),
#     }
# )

# plot_rate_ms_ssim(
#     out_file='paper/celebahq256_top_rate_ms_ssim.pdf',
#     results={
#         'Joint'         : glob.glob('checkpoints/celebahq256_vqvae_joint_[1234]bit/eval_*.pt'),
#         'Distributed'   : glob.glob('checkpoints/celebahq256_vqvae_dist_[1234]bit/eval_*.pt'),
#         'Separate'      : glob.glob('checkpoints/celebahq256_vqvae_separate_[1234]bit/eval_*.pt'),
#     }
# )

plot_bpp_psnr(
    out_file='paper/celebahq256_top_bpp_psnr.pdf',
    results={
        'Distributed'   : glob.glob('checkpoints/celebahq256_cvqvae_top_dist_[1234]bit/eval_*.pt'),
        'Joint'         : glob.glob('checkpoints/celebahq256_cvqvae_top_joint_[1234]bit/eval_*.pt'),
        'Separate'      : glob.glob('checkpoints/celebahq256_cvqvae_top_separate_[1234]bit/eval_*.pt'),
    },
)

plot_bpp_psnr(
    out_file='paper/celebahq256_top_bpp_psnr_ours_vs_ndic.pdf',
    results={
        'Ours'   : glob.glob('checkpoints/celebahq256_cvqvae_top_dist_[1234]bit/eval_*.pt'),
    },
    others={
        'NDIC'          : NDIC_RESULT_FILES,
    }
)

plot_bpp_ms_ssim(
    out_file='paper/celebahq256_top_bpp_ms_ssim.pdf',
    results={
        'Joint'         : glob.glob('checkpoints/celebahq256_cvqvae_top_joint_[1234]bit/eval_*.pt'),
        'Distributed'   : glob.glob('checkpoints/celebahq256_cvqvae_top_dist_[1234]bit/eval_*.pt'),
        'Separate'      : glob.glob('checkpoints/celebahq256_cvqvae_top_separate_[1234]bit/eval_*.pt'),
    },
    others={
        # 'NDIC'          : glob.glob('ndic_exp/eval_results/bmshj18.celebahq.ms_ssim.lambda_*/eval_result.csv'),
        'NDIC'          : NDIC_RESULT_FILES,
    }
)

