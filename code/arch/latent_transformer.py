import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from code import vqvae
from code.utils import HParams, get_param_count


_HP_BASE = HParams(
    seed                = 1234,

    # parameters to be specified at model creation
    codebook_bits       = None,
    latent_shape        = None,

    # Transformer decoder architecture
    dim                 = 128,
    n_heads             = 4,
    n_blocks            = 4,
    dk                  = 128 // 4,     # dk = dv = dim // n_heads
    dv                  = 128 // 4,
    dff                 = 128 * 4,      # dimension of inner layer of feed forward networks (dim * 4)
    
    # Training
    beta1               = 0.9,
    beta2               = 0.999,
    eps                 = 1e-8,
    epochs              = 100,
    bs                  = 12,
    lr                  = 3e-4,
    lr_warmup           = 2000,
    clip_grad_norm      = 100,

    # Monitoring
    print_freq          = 10,
    log_freq            = 50,
    sample_freq         = 25,
    ckpt_freq           = 50,
)


class DecoderBlock(nn.Module):
    def __init__(self, dim, dk, dv, dff, n_heads, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.multi_head_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, bias=True, add_bias_kv=False, batch_first=True)
        self.layernorm1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dff, bias=True),
            nn.GELU(),
            nn.Linear(dff, dim, bias=True)
        )
        self.layernorm2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        out, _ = self.multi_head_attn(x, x, x, attn_mask=mask)
        x = self.layernorm1(self.dropout(out) + x)
        out = self.feedforward(x)
        out = self.layernorm2(self.dropout(out) + x)
       
        return out


class PositionalEncoder(nn.Module):
    def __init__(self, dim, latent_shape, dropout=0.1):
        super(PositionalEncoder, self).__init__()

        self.length = latent_shape[1] * latent_shape[2] + 1     # add one additional embedding for the the <start> token
        pos_enc = torch.zeros(self.length, dim)
        pos = torch.arange(0, self.length).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pos_enc[:, 0::2] = torch.sin(pos * div)
        pos_enc[:, 1::2] = torch.cos(pos * div)
        pos_enc.requires_grad = False
        self.register_buffer('pos_enc', pos_enc)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x + self.pos_enc)


class LearnedEmbedding(nn.Module):
    def __init__(self, codebook_size, dim=128):
        super(LearnedEmbedding, self).__init__()

        self.emb = nn.Embedding(codebook_size + 1, dim)     # add one additional embedding for the <start> token
        self.dim = dim

    def forward(self, x):
        return self.emb(x)


class LatentTransformer(nn.Module):
    def __init__(self, *,
                 latent_shape,  # latent image shape, e.g., (1, 16, 32)
                 codebook_bits: int,
                 dropout=0.1,
                 **kwargs):
        super(LatentTransformer, self).__init__()

        self.hp = _HP_BASE.clone()
        self.hp.latent_shape = latent_shape
        self.hp.codebook_bits = codebook_bits

        for k, v in kwargs.items():
            assert k in self.hp, f'Invalid hparam {k} given'
            if v != self.hp[k]:
                print(f'Overriding hparam {k}: {v} (default: {self.hp[k]})')
                self.hp[k] = v

        self.embedding = LearnedEmbedding(2 ** self.hp.codebook_bits, self.hp.dim)
        self.encoder = PositionalEncoder(self.hp.dim, self.hp.latent_shape, dropout)
        self.blocks = nn.ModuleList()
        self.linear = nn.Linear(self.hp.dim, 2 ** self.hp.codebook_bits, bias=True)
        self.mask = torch.triu(torch.ones(latent_shape[1] * latent_shape[2] + 1, latent_shape[1] * latent_shape[2] + 1, dtype=bool), diagonal=1).to('cuda')
        
        for _ in range(self.hp.n_blocks):
            self.blocks.append(DecoderBlock(self.hp.dim, self.hp.dk, self.hp.dv, self.hp.dff, self.hp.n_heads, dropout))

    def forward(self, x):
        x = torch.cat((int(2 ** self.hp.codebook_bits) * torch.ones(x.size()[0], 1, dtype=int).to('cuda'), x), dim=1)     # prepend <start> token to each batch sequence, corresponding to the last embedding vector
        out = self.embedding(x)
        out = self.encoder(out)
        for block in self.blocks:
            out = block(out, self.mask)

        out = out[:, :-1, :]        # we have latent_shape[1] * latent_shape[2] dimensions, so we don't need to learn the distribution for some (latent_shape[1] * latent_shape[2] + 1)-th dimension and we can remove it
        logits = self.linear(out)
        prob = F.softmax(logits, dim=-1)

        return logits, prob

