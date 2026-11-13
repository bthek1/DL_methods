# Model Design Types - Overview

A map of the major neural network families: what they are, what they are good at, and how to choose between them. Each type has its own notebook (01-10) with theory and runnable PyTorch examples.

Author

Benedict Thekkel

## How to use this series

Each notebook is self-contained and follows the same shape: intuition, the core math, a from-scratch implementation, the idiomatic PyTorch version, a minimal training loop, and a strengths/weaknesses summary.

| \#  | Type        | One-line role                            |
|-----|-------------|------------------------------------------|
| 01  | FNN / MLP   | Dense mapping of fixed-size vectors      |
| 02  | CNN         | Spatial / grid data (images)             |
| 03  | RNN         | Sequences via a recurrent hidden state   |
| 04  | LSTM        | Long sequences with gated memory         |
| 05  | GRU         | Leaner gated recurrence                  |
| 06  | Autoencoder | Unsupervised compression / generation    |
| 07  | GAN         | Adversarial generative modeling          |
| 08  | Transformer | Attention over sequences and patches     |
| 09  | RBM         | Energy-based stochastic feature learning |
| 10  | DBN         | Stacked RBMs with greedy pretraining     |

------------------------------------------------------------------------

## A mental model of the families

Three broad lineages cover almost everything below:

- Feedforward (FNN, CNN, Transformer): one pass input to output, no recurrence. CNNs add spatial inductive bias; Transformers add attention.
- Recurrent (RNN, LSTM, GRU): carry a hidden state across time steps. Largely superseded by Transformers for NLP, still useful for streaming / low-latency / small sequence tasks.
- Generative / energy-based (Autoencoder, GAN, RBM, DBN): learn the data distribution rather than a label map. Modern generation leans on VAEs, GANs, and diffusion; RBM/DBN are mostly of historical importance.

------------------------------------------------------------------------

## Choosing a model by data type

| Your data | Start with | Why |
|----|----|----|
| Tabular / fixed vectors | FNN (MLP) | No spatial or temporal structure to exploit |
| Images / grids | CNN, or ViT (Transformer) | Local receptive fields and weight sharing |
| Text / tokens | Transformer | Parallel, long-range attention; LSTM/GRU if tiny or streaming |
| Time series | LSTM / GRU / Temporal CNN / Transformer | Depends on horizon and latency |
| Unlabeled, want features | Autoencoder / RBM | Reconstruction or energy-based pretraining |
| Want to synthesize samples | GAN / VAE / Diffusion | Learn and sample the data distribution |

------------------------------------------------------------------------

## Parameter and compute intuition

``` python
import torch.nn as nn

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# Same input width (64), same hidden (128): compare a few building blocks
mlp  = nn.Linear(64, 128)                 # dense
rnn  = nn.RNN(64, 128, batch_first=True)  # 1 gate
gru  = nn.GRU(64, 128, batch_first=True)  # 3 gates
lstm = nn.LSTM(64, 128, batch_first=True) # 4 gates

for name, m in [('Linear', mlp), ('RNN', rnn), ('GRU', gru), ('LSTM', lstm)]:
    print(f'{name:7} params = {count_params(m):,}')
# Rough ratio RNN:GRU:LSTM is about 1:3:4 (gate count drives parameter count)
```

    Linear  params = 8,320
    RNN     params = 24,832
    GRU     params = 74,496
    LSTM    params = 99,328

## Rules of thumb

- Start simple. A well-tuned MLP or CNN is a strong baseline before reaching for attention.
- Match the inductive bias to the data: convolution for locality, attention for global context, recurrence for strict ordering with small state.
- For generation today, prefer VAE / GAN / diffusion over RBM / DBN; the latter two are best understood as the historical bridge (Hinton 2006) that revived deep learning.
- Transformers dominate at scale but cost O(n^2) in sequence length; recurrent models are O(n) memory and shine for long streaming inputs.

------------------------------------------------------------------------

## Next

Open the numbered notebooks in order, or jump straight to the family you need. The `Model Design/` subfolder contains end-to-end build-from-scratch case studies (MNIST MLP, CNN, ResNet18).

------------------------------------------------------------------------

Back to top
