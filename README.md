# Setup
## Installation
We use [Anaconda](https://www.anaconda.com/products/individual) for managing Python environment.
```shell
conda env create --file environment.yml
conda activate neural_dsc
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
HOROVOD_GPU_OPERATIONS=NCCL pip install --upgrade --no-cache-dir "horovod[pytorch]"
```

----

# Experiments

## Activate the environment
```shell
conda activate neural_dsc
```

## Experiment 1: KITTI Stereo
KITTI Stereo experiments (as shown in Figure 2) can be reproduced by executing the run-kitti.sh script. 

NOTE: Please contact the corresponding author at `acnagle@utexas.edu` for more information on gathering the KITTI dataset.

## Experiment 2: VQ-VAE comparison with DISCUS
Training the VQ-VAE on Bernoulli sequences (as shown in Figure 6) can be reproduced by executing the `run-discus.sh` script. Note that the LDPC code is implemented in MATLAB. The script to gather the LDPC results can be found in the `baseline/nonlearning/` directory. This MATLAB script saves its results into a `iid_flip.csv` file.

## Experiment 3: CelebA-HQ

### Prepare data (CelebA-HQ 256x256)
Download `celeba-tfr.tar` inside `data/` directory, then run the following command:
```shell
python run_top.py prep celebahq256
```

### Train VQ-VAE
Repeat the following with different `--codebook_bits` argument to control the total rate.
```shell
# Joint VQ-VAE
horovodrun -n 2 python -O run_top.py train --dataset celebahq256 --arch vqvae_top_8x --ch_latent 1 \
  --root_dir checkpoints/celeba256_vqvae_joint_4bit --dec_si True --enc_si True  --codebook_bits 4

# Distributed VQ-VAE
horovodrun -n 2 python -O run_top.py train --dataset celebahq256 --arch vqvae_top_8x --ch_latent 1 \
  --root_dir checkpoints/celeba256_vqvae_dist_4bit --dec_si True --enc_si False --codebook_bits 4

# Separate VQ-VAE
horovodrun -n 2 python -O run_top.py train --dataset celebahq256 --arch vqvae_top_8x --ch_latent 1 \
  --root_dir checkpoints/celeba256_vqvae_separate_4bit --dec_si False --enc_si False --codebook_bits 4
```

### Evaluate VQ-VAE
```shell
horovodrun -n 2 python run_top.py eval --batch_size 250 \
checkpoints/celebahq256_vqvae_{joint,dist,separate}_4bit/ckpt_ep=020_step=0016880.pt
```

## Plot rate-distortion curves from eval results for experiments 1 through 3
All generated plots will be stored in the folder `paper/`.
```shell
python plot_rd_curves.py
```

----

## Experiment 4: Distributed SGD

### Prepare data
```shell
# Following command may take a while to finish due to slow download speed.
python run_mnist_grad.py prep mnist
```

### Gather gradients
```shell
python -O run_mnist_grad.py gather_gradients --out_dir checkpoints/mnist_grad_data
```

### Train VQ-VAE
```shell
# Joint VQ-VAE
python -O run_mnist_grad.py train_vqvae --grad_dump checkpoints/mnist_grad_data/grads.pt --d_latent 40 --codebook_bits 8 \
  --enc_si True  --dec_si True  --root_dir checkpoints/mnist_grad_vqvae_joint_40d_8bits

# Distributed VQ-VAE
python -O run_mnist_grad.py train_vqvae --grad_dump checkpoints/mnist_grad_data/grads.pt --d_latent 40 --codebook_bits 8 \
  --enc_si False --dec_si True  --root_dir checkpoints/mnist_grad_vqvae_dist_40d_8bits

# Separate VQ-VAE
python -O run_mnist_grad.py train_vqvae --grad_dump checkpoints/mnist_grad_data/grads.pt --d_latent 40 --codebook_bits 8 \
  --enc_si False --dec_si False --root_dir checkpoints/mnist_grad_vqvae_separate_40d_8bits
```

### Evaluate
```shell
for seed in $(seq 1 20); do
  python run_mnist_grad.py eval checkpoints/mnist_grad_vqvae_{joint,dist,separate}_40d_8bits/ckpt_ep=500_step=0391000.pt --seed $seed;
done
```

### Plot
```shell
python run_mnist_grad.py plot checkpoints/mnist_grad_vqvae_{joint,dist,separate}_40d_8bits/ckpt_ep=500_step=0391000.pt \
  --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --out_dir paper --labels Joint,Distributed,Separate
```
