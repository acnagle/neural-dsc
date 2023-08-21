import argparse
import fire
import glob
import os
from typing import Dict

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
import seaborn as sns; sns.set(context='paper', style='white', font_scale=1.6, font='Times New Roman')
import seaborn as sns

import numpy as np
import torch

from benchmark.ndic import ndic_results

_MARKERS = ['o', '^', 's', 'x', 'm']


def _plot_points(ax, points, **kwargs):
    xs, ys = zip(*points)
    ax.plot(xs, ys, **kwargs)


def transform_ssim(points):
    for i in range(len(points)):
        points[i] = (points[i][0], 1 - 10 ** (-1 * points[i][1] / 10))
    return points


def plot_bpp_psnr(out_file: str, results: Dict):
    rd_points = []
    rd_points_prior = []
    for label, result in results.items():
        if os.path.isdir(result):
            result = glob.glob(os.path.join(result, 'eval_*.pt'))
        else:
            ckpt = result
            result = [ckpt]
            result.append(ckpt[:-3] + '_latent_prior.pt')
        dd = torch.load(result[0] if 'latent_prior' not in result[0] else result[1])
        dd_prior = torch.load(result[1] if 'latent_prior' in result[1] else result[0])
        rd_points.append((dd['bpp'], dd['all_psnr'].mean()))
        rd_points_prior.append((dd_prior['bpp'], dd_prior['all_psnr'].mean()))

    print(f'Loaded {len(rd_points)} R-D points.')

    plt.clf()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1,1,1)

    _plot_points(ax, ndic_results.KITTI_PSNR['dsin'], label='DSIN', markersize=8, marker='s', color='tab:green')
    _plot_points(ax, ndic_results.KITTI_PSNR['balle17'], label='NDIC (Ballé17)', markersize=8, marker='x', color='tab:red')
    _plot_points(ax, ndic_results.KITTI_PSNR['balle18'], label='NDIC (Ballé18)', markersize=8, marker='^', color='tab:orange')
    _plot_points(ax, rd_points, label='Ours', markersize=8, marker='o', color='tab:blue', linestyle='dashed')
    _plot_points(ax, rd_points_prior, label='Ours+Latent Prior', markersize=8, marker='o', color='tab:blue')

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Rate (bpp)')
    ax.set_ylabel('PSNR')
    ax.set_ylim(18.0, 30)
    ax.set_xlim(-0.01, 0.5)
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def plot_bpp_ssim(out_file: str, results: Dict):
    rd_points = []
    rd_points_prior = []
    for label, result in results.items():
        if os.path.isdir(result):
            result = glob.glob(os.path.join(result, 'eval_*.pt'))
        else:
            ckpt = result
            result = [ckpt]
            result.append(ckpt[:-3] + '_latent_prior.pt')
        dd = torch.load(result[0] if 'latent_prior' not in result[0] else result[1])
        dd_prior = torch.load(result[1] if 'latent_prior' in result[1] else result[0])
        rd_points.append((dd['bpp'], dd['all_ms_ssim'].mean()))
        rd_points_prior.append((dd_prior['bpp'], dd_prior['all_ms_ssim'].mean()))

    print(f'Loaded {len(rd_points)} R-D points.')

    plt.clf()
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1,1,1)
    # xticks = set()

    _plot_points(ax, transform_ssim(ndic_results.KITTI_SSIM['dsin']), label='DSIN', markersize=8, marker='s', color='tab:green')
    _plot_points(ax, transform_ssim(ndic_results.KITTI_SSIM['balle17']), label='NDIC (Ballé17)', markersize=8, marker='x', color='tab:red')
    _plot_points(ax, transform_ssim(ndic_results.KITTI_SSIM['balle18']), label='NDIC (Ballé18)', markersize=8, marker='^', color='tab:orange')
    _plot_points(ax, transform_ssim(ndic_results.KITTI_SSIM['ndic-cam']), label='NDIC-CAM', markersize=8, marker='d', color='black')
    _plot_points(ax, rd_points, label='Ours', markersize=8, marker='o', color='tab:blue', linestyle='dashed')
    _plot_points(ax, rd_points_prior, label='Ours+Latent Prior', markersize=8, marker='o', color='tab:blue')

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Rate (bpp)')
    ax.set_ylabel('MS-SSIM')
    ax.set_xlim(-0.01, 0.4)
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


def plot_rate_mse(*, out_file: str, results: Dict):
    assert all(all(os.path.isfile(fn) for fn in fns) for fns in results.values())
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(1,1,1)
    xticks = set()

    for i, (label, files) in enumerate(results.items()):
        rate_mse = []
        for fn in files:
            res = torch.load(fn)
            if res['rate'] > 2048:
                continue
            rate_mse.append((res['rate'], res['mse'].mean().item()))
            xticks.add(res['rate'])
        rate_mse = sorted(rate_mse, key=lambda x: x[0])
        ax.plot(*zip(*rate_mse), label=label, markersize=5, linewidth=1, marker=_MARKERS[i])

    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    ax.set_xlabel('Rate (bits)')
    ax.set_ylabel('MSE')
    ax.set_xticks(list(xticks))
    ax.set_xticklabels(list(xticks))
    ax.legend(loc='best')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)



RESULTS_OURS = {
    # bpp = 0.015625
    'big-1bit': 'checkpoints/kitti_stereo_big_dist_1bit',

    # bpp = 0.03125
    'big2-2bit': 'checkpoints/kitti_stereo_big2_dist_2bit',

    # bpp = 0.046875
    'big-3bit': 'checkpoints/kitti_stereo_big_dist_3bit/eval_ep=500_step=0049500_jay.pt',
#    'big-3bit': 'checkpoints/kitti_stereo_big_dist_3bit',

    # bpp = 0.125
    'big-8bit': 'checkpoints/kitti_stereo_big2_dist_8bit/eval_ep=600_step=0059400_jay.pt',
#    'big-8bit': 'checkpoints/kitti_stereo_big2_dist_8bit',

    # bpp = 0.1875
    '4x-3bit': 'checkpoints/kitti_stereo_4x_dist_3bit/eval_ep=1000_step=0099000_jay.pt',
#    '4x-3bit': 'checkpoints/kitti_stereo_4x_dist_3bit',

    # bpp = 0.250
    '4x-4bit': 'checkpoints/kitti_stereo_4x_dist_4bit',

    # bpp = 0.375
    '4x-6bit': 'checkpoints/kitti_stereo_4x_dist_6bit',

    # bpp = 0.75
    '2x-3bit': 'checkpoints/kitti_stereo_2x_dist_3bit',

    # bpp = 1
    '2x-4bit': 'checkpoints/kitti_stereo_2x_dist_4bit',
}


def print_model_param_count(results: Dict):
    print('Model params:')
    for label, result in results.items():
        if os.path.isdir(result):
            ckpt = sorted(glob.glob(os.path.join(result, 'ckpt_*.pt')))[-1]
            eval_result = glob.glob(os.path.join(result, 'eval_*.pt'))
        else:
            eval_result = [result]
            eval_result.append(result[:-3] + '_latent_prior.pt')
            ckpt_split = result.split('/')
            ckpt = '/'.join(ckpt_split[:-1]) + '/ckpt' + ckpt_split[-1][4:]

        ckpt_prior = sorted(glob.glob(os.path.join('/'.join(ckpt.split('/')[:-1]), 'learned_prior/ckpt_*.pt')))[-1]
        print(ckpt_prior)
        ckpt_prior = torch.load(ckpt_prior)
        model_state_dict = ckpt_prior['model_state_dict']
        num_params_prior = sum(np.prod(p.shape) for p in model_state_dict.values())
            
        ckpt = torch.load(ckpt)
        dd = torch.load(eval_result[0] if 'latent_prior' not in eval_result[0] else eval_result[1])
        dd_prior = torch.load(eval_result[1] if 'latent_prior' in eval_result[1] else eval_result[0])
        model_state_dict = ckpt['model_state_dict']
        num_params = sum(np.prod(p.shape) for p in model_state_dict.values())
        print(f'  -> {label:10s}: bpp {dd["bpp"]:.4f} bpp_with_prior {dd_prior["bpp"]:.4f} psnr {dd["all_psnr"].mean():.3f} ms-ssim {dd["all_ms_ssim"].mean():.3f} params {num_params} total_params_prior {num_params_prior} total_params_with_prior {num_params + num_params_prior}')


if __name__ == '__main__':
    plot_bpp_psnr('paper/kitti_stereo_bpp_psnr.pdf', RESULTS_OURS)
    plot_bpp_ssim('paper/kitti_stereo_bpp_ssim.pdf', RESULTS_OURS)
    print_model_param_count(RESULTS_OURS)
