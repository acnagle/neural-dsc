#!/usr/bin/env bash

# The following 9 models correspond to each rate-distortion point in Figure 2
# Train the VQ-VAE models
python -O run_kitti_stereo_single.py train --arch kitti_stereo_big2 --root_dir checkpoints/kitti_stereo_big_dist_1bit --dec_si True --enc_si False --codebook_bits 1 --ch_latent 1 --decay 0.995 --learning_rate 2e-4 --ckpt_freq 25 --sample_freq 5
python -O run_kitti_stereo_single.py train --arch kitti_stereo_big2 --root_dir checkpoints/kitti_stereo_big2_dist_2bit --dec_si True --enc_si False --codebook_bits 2 --ch_latent 1 --decay 0.995 --learning_rate 2e-4 --ckpt_freq 25 --sample_freq 5
python -O run_kitti_stereo_single.py train --arch kitti_stereo_big2 --root_dir checkpoints/kitti_stereo_big_dist_3bit --dec_si True --enc_si False --codebook_bits 3 --ch_latent 1 --decay 0.995 --learning_rate 2e-4 --ckpt_freq 25 --sample_freq 5
python -O run_kitti_stereo_single.py train --arch kitti_stereo_big2 --root_dir checkpoints/kitti_stereo_big2_dist_8bit --dec_si True --enc_si False --codebook_bits 8 --ch_latent 1 --decay 0.995 --learning_rate 2e-4 --ckpt_freq 25 --sample_freq 5
python -O run_kitti_stereo_single.py train --arch kitti_stereo_4x --root_dir checkpoints/kitti_stereo_4x_dist_3bit --dec_si True --enc_si False --codebook_bits 3 --ch_latent 1 -- ckpt_freq 25 --sample_freq 5
python -O run_kitti_stereo_single.py train --arch kitti_stereo_4x --root_dir checkpoints/kitti_stereo_4x_dist_4bit --dec_si True --enc_si False --codebook_bits 4 --ch_latent 1 -- ckpt_freq 25 --sample_freq 5
python -O run_kitti_stereo_single.py train --arch kitti_stereo_4x --root_dir checkpoints/kitti_stereo_4x_dist_6bit --dec_si True --enc_si False --codebook_bits 6 --ch_latent 1 -- ckpt_freq 25 --sample_freq 5 --decay 0.995 --learning_rate 2e-4
python -O run_kitti_stereo_single.py train --arch kitti_stereo_2x --root_dir checkpoints/kitti_stereo_2x_dist_3bit --dec_si True --enc_si False --codebook_bits 3 --ch_latent 1 -- ckpt_freq 25 --sample_freq 5 --decay 0.995 --learning_rate 2e-4
python -O run_kitti_stereo_single.py train --arch kitti_stereo_2x --root_dir checkpoints/kitti_stereo_2x_dist_4bit --dec_si True --enc_si False --codebook_bits 4 --ch_latent 1 -- ckpt_freq 25 --sample_freq 5 --decay 0.995 --learning_rate 2e-4

# Train the latent prior models
python -O run_kitti_stereo_single.py train_prior --vqvae-ckpt checkpoints/kitti_stereo_2x_dist_3bit/ckpt_ep=1000_step=0099000.pt --dim 128 --n_blocks 2 --n_heads 2 --dk 32 --dv 32 --batch_size 5
python -O run_kitti_stereo_single.py train_prior --vqvae-ckpt checkpoints/kitti_stereo_2x_dist_4bit/ckpt_ep=1000_step=0099000.pt --dim 128 --n_blocks 2 --n_heads 2 --dk 32 --dv 32 --batch_size 5

python -O run_kitti_stereo_single.py train_prior --vqvae-ckpt checkpoints/kitti_stereo_4x_dist_3bit/ckpt_ep=1000_step=0099000.pt --dim 128 --n_blocks 4 --n_heads 4 --dk 32 --dv 32 --batch_size 16
python -O run_kitti_stereo_single.py train_prior --vqvae-ckpt checkpoints/kitti_stereo_4x_dist_4bit/ckpt_ep=1000_step=0099000.pt --dim 128 --n_blocks 4 --n_heads 4 --dk 32 --dv 32 --batch_size 16
python -O run_kitti_stereo_single.py train_prior --vqvae-ckpt checkpoints/kitti_stereo_4x_dist_6bit/ckpt_ep=1000_step=0099000.pt --dim 128 --n_blocks 4 --n_heads 4 --dk 32 --dv 32 --batch_size 16

python -O run_kitti_stereo_single.py train_prior --vqvae-ckpt checkpoints/kitti_stereo_big_dist_1bit/ckpt_ep=1000_step=0099000.pt --dim 128 --n_blocks 4 --n_heads 4 --dk 32 --dv 32 --batch_size 128
python -O run_kitti_stereo_single.py train_prior --vqvae-ckpt checkpoints/kitti_stereo_big_dist_3bit/ckpt_ep=500_step=0049500.pt --dim 128 --n_blocks 4 --n_heads 4 --dk 32 --dv 32 --batch_size 128

python -O run_kitti_stereo_single.py train_prior --vqvae-ckpt checkpoints/kitti_stereo_big2_dist_2bit/ckpt_ep=1000_step=0099000.pt --dim 128 --n_blocks 4 --n_heads 4 --dk 32 --dv 32 --batch_size 128
python -O run_kitti_stereo_single.py train_prior --vqvae-ckpt checkpoints/kitti_stereo_big2_dist_8bit/ckpt_ep=600_step=0059400.pt --dim 128 --n_blocks 4 --n_heads 4 --dk 32 --dv 32 --batch_size 128

## Evaluate & Plot
# Evaluate
python run_kitti_stereo_single.py eval --ckpts checkpoints/kitti_stereo_big_dist_1bit,checkpoints/kitti_stereo_big2_dist_2bit,checkpoints/kitti_stereo_big_dist_3bit,checkpoints/kitti_stereo_big2_dist_8bit,checkpoints/kitti_stereo_4x_dist_3bit,checkpoints/kitti_stereo_4x_dist_4bit,checkpoints/kitti_stereo_4x_dist_6bit,checkpoints/kitti_stereo_2x_dist_3bit,checkpoints/kitti_stereo_2x_dist_4bit --batch_size 16 --use_prior False --overwrite True

# Evaluate with Latent Prior
python run_kitti_stereo_single.py eval --ckpts checkpoints/kitti_stereo_big_dist_1bit,checkpoints/kitti_stereo_big2_dist_2bit,checkpoints/kitti_stereo_big_dist_3bit,checkpoints/kitti_stereo_big2_dist_8bit,checkpoints/kitti_stereo_4x_dist_3bit,checkpoints/kitti_stereo_4x_dist_4bit,checkpoints/kitti_stereo_4x_dist_6bit,checkpoints/kitti_stereo_2x_dist_3bit,checkpoints/kitti_stereo_2x_dist_4bit --batch_size 16 --use_prior True --overwrite True

# Generate rate-distortion plots
python plot_rd_kitti.py
