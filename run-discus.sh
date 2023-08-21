#!/usr/bin/env bash

# Train our VQ-VAE model on i.i.d. Bernoulli messages
python run_discus.py train_all --root_dir ./baseline --rates 1 --noise_probs 0.0,0.0006,0.0012,0.0018,0.0024,0.003,0.0036,0.0042,0.0048,0.0054,0.006,0.0066,0.0072,0.0078,0.0084,0.009,0.0096,0.0102,0.0108,0.0114,0.015,0.0204,0.0252,0.03 --seq_lens 648 --d_latents 162
