# Transformer Networks

Use self-attention mechanisms to handle sequential data without the recurrence found in RNNs. They can capture long-range dependencies and parallelize training more efficiently.

Author

Benedict Thekkel

## Intuition: Attention Replaces Recurrence

Recurrent networks (RNN, LSTM, GRU) process a sequence one step at a time. The hidden state at position `t` depends on the hidden state at `t-1`, so the computation is inherently sequential and cannot be parallelized across the time axis. Long-range dependencies must survive many steps of repeated multiplication, which leads to vanishing or exploding gradients.

The Transformer replaces recurrence with *self-attention*. Every position can look directly at every other position in a single operation. This gives three big wins:

- **Parallelism**: all positions are processed at once via matrix multiplies, so a GPU can fill an entire sequence in one shot during training.
- **Long-range dependencies**: the path length between any two tokens is O(1) instead of O(n), so distant tokens interact directly.
- **Content-based routing**: attention weights are data-dependent, so the model learns *which* tokens matter for each query rather than relying on a fixed window.

The cost is that full attention is O(n^2) in sequence length, which we revisit later.

------------------------------------------------------------------------

## Scaled Dot-Product Attention

Attention maps a set of queries `Q` against keys `K` and values `V`. Each query is compared to every key with a dot product, the scores are normalized with softmax, and the result weights the values:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Shapes: with `Q` of shape `(n_q, d_k)`, `K` of shape `(n_k, d_k)`, and `V` of shape `(n_k, d_v)`, the score matrix `Q K^T` is `(n_q, n_k)` and the output is `(n_q, d_v)`.

The scaling factor $1/\sqrt{d_k}$ matters: for large `d_k` the dot products grow in magnitude, pushing softmax into regions with tiny gradients. Dividing by $\sqrt{d_k}$ keeps the variance of the scores near 1.

An optional *mask* sets disallowed score positions to a large negative number before the softmax so they receive zero weight (used for padding and for causal decoding).

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q: (batch, heads, n_q, d_k)
    # K: (batch, heads, n_k, d_k)
    # V: (batch, heads, n_k, d_v)
    # mask: broadcastable to (batch, heads, n_q, n_k), True = keep, False = block
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, heads, n_q, n_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)            # (batch, heads, n_q, n_k)
    out = torch.matmul(attn, V)                 # (batch, heads, n_q, d_v)
    return out, attn

# quick shape check
q = torch.randn(2, 4, 10, 16)  # batch=2, heads=4, seq=10, d_k=16
k = torch.randn(2, 4, 10, 16)
v = torch.randn(2, 4, 10, 32)  # d_v can differ from d_k
out, attn = scaled_dot_product_attention(q, k, v)
print(out.shape, attn.shape)   # (2, 4, 10, 32) (2, 4, 10, 10)
```

    torch.Size([2, 4, 10, 32]) torch.Size([2, 4, 10, 10])

## Multi-Head Attention

A single attention function averages information into one representation subspace. *Multi-head attention* runs `h` attention operations in parallel, each with its own learned projections, so different heads can attend to different relationships (syntax, coreference, position, etc.).

Given model dimension `d_model` and `h` heads, each head works in dimension `d_k = d_model / h`. We project the input to `Q`, `K`, `V`, reshape into heads, run scaled dot-product attention, concatenate the heads, and apply a final output projection.

------------------------------------------------------------------------

``` python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # x: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        # query/key/value: (batch, seq, d_model)
        Q = self.split_heads(self.w_q(query))  # (batch, heads, seq_q, d_k)
        K = self.split_heads(self.w_k(key))    # (batch, heads, seq_k, d_k)
        V = self.split_heads(self.w_v(value))  # (batch, heads, seq_k, d_k)
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # broadcast over batch and heads
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V)            # (batch, heads, seq_q, d_k)
        # concat heads -> (batch, seq_q, d_model)
        batch, _, seq_q, _ = ctx.shape
        ctx = ctx.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)
        return self.w_o(ctx)                   # (batch, seq_q, d_model)

mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)
print(mha(x, x, x).shape)  # (2, 10, 64) -- self-attention: query=key=value=x
```

    torch.Size([2, 10, 64])

## Positional Encoding

Attention is *permutation invariant*: it treats the input as a set, so without extra information it cannot tell token order. We inject order with a *positional encoding* added to the token embeddings.

The original Transformer uses fixed sinusoidal encodings. For position `pos` and dimension index `i`:

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Sinusoids of geometrically spaced frequencies let the model attend to relative positions: for any fixed offset `k`, `PE(pos+k)` is a linear function of `PE(pos)`. Sinusoidal encodings also extrapolate to sequence lengths not seen in training. Modern models often use *learned* positional embeddings or rotary embeddings (RoPE) instead, but the sinusoidal version is the classic baseline.

------------------------------------------------------------------------

``` python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)                  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                   # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # not a parameter, but moves with .to(device)

    def forward(self, x):
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

pos = PositionalEncoding(d_model=64)
print(pos(torch.randn(2, 10, 64)).shape)  # (2, 10, 64)
```

    torch.Size([2, 10, 64])

## The Transformer Block (Encoder)

A Transformer block stacks two sublayers, each wrapped in a residual connection and layer normalization:

1.  Multi-head self-attention.
2.  A position-wise feedforward network (two linear layers with a non-linearity, usually with an inner dimension 4x larger than `d_model`).

There are two conventions for where the LayerNorm goes:

- **Post-norm** (original paper): `x = LayerNorm(x + Sublayer(x))`. Harder to train deep; needs learning-rate warmup.
- **Pre-norm** (GPT-2 and most modern models): `x = x + Sublayer(LayerNorm(x))`. More stable gradients, trains deep stacks reliably.

The residual connections let gradients flow directly, and LayerNorm keeps activations well scaled. The block below is pre-norm.

------------------------------------------------------------------------

``` python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)  # (batch, seq, d_model)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # pre-norm: normalize before each sublayer, add residual after
        a = self.norm1(x)
        x = x + self.dropout(self.attn(a, a, a, mask))  # (batch, seq, d_model)
        f = self.norm2(x)
        x = x + self.dropout(self.ff(f))                # (batch, seq, d_model)
        return x

block = EncoderBlock(d_model=64, num_heads=8, d_ff=256)
print(block(torch.randn(2, 10, 64)).shape)  # (2, 10, 64)
```

    torch.Size([2, 10, 64])

## Encoder vs Decoder and Causal Masking

The original Transformer is an *encoder-decoder*:

- The **encoder** reads the whole input with bidirectional self-attention (every token sees every other token).
- The **decoder** generates output autoregressively. It uses *masked* self-attention so that position `t` can only attend to positions `<= t` (it must not peek at future tokens it has not generated yet), plus *cross-attention* over the encoder output.

The causal (look-ahead) mask is a lower-triangular matrix: position `i` may attend to key `j` only when `j <= i`. During training this lets us compute the loss for all positions in parallel while preserving the autoregressive property.

------------------------------------------------------------------------

``` python
def causal_mask(seq_len):
    # lower-triangular ones; True/1 = allowed, 0 = blocked (future)
    mask = torch.tril(torch.ones(seq_len, seq_len))  # (seq, seq)
    return mask.bool()

m = causal_mask(5)
print(m.int())
# tensor([[1, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0],
#         [1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1]])

# use it in self-attention so each position only sees itself and earlier positions
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 5, 64)
out = mha(x, x, x, mask=causal_mask(5))
print(out.shape)  # (2, 5, 64)
```

    tensor([[1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]], dtype=torch.int32)
    torch.Size([2, 5, 64])

## Using PyTorch Built-ins

You rarely re-implement attention in production. PyTorch ships optimized modules:

- `nn.MultiheadAttention`: a single multi-head attention layer. Note `batch_first=True` to use `(batch, seq, feature)` ordering; otherwise it expects `(seq, batch, feature)`.
- `nn.TransformerEncoderLayer`: one full encoder block (attention + FFN + norms).
- `nn.TransformerEncoder`: a stack of N encoder layers.

There are also `nn.TransformerDecoderLayer`, `nn.TransformerDecoder`, and the full `nn.Transformer`. For masking, `src_key_padding_mask` blocks padding tokens and a float/bool `mask` enforces causality.

------------------------------------------------------------------------

``` python
# single attention layer
attn = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
x = torch.randn(2, 10, 64)  # (batch, seq, embed)
out, weights = attn(x, x, x)
print(out.shape, weights.shape)  # (2, 10, 64) (2, 10, 10)

# one encoder block
layer = nn.TransformerEncoderLayer(
    d_model=64, nhead=8, dim_feedforward=256,
    dropout=0.1, activation='gelu', batch_first=True, norm_first=True,  # norm_first => pre-norm
)
encoder = nn.TransformerEncoder(layer, num_layers=6)
print(encoder(x).shape)  # (2, 10, 64)

# causal mask for an autoregressive decoder (True = block in this API)
causal = nn.Transformer.generate_square_subsequent_mask(10)  # (10, 10) float -inf above diag
print(causal.shape)
```

    torch.Size([2, 10, 64]) torch.Size([2, 10, 10])

    /tmp/ipykernel_301962/767861168.py:12: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
      encoder = nn.TransformerEncoder(layer, num_layers=6)

    torch.Size([2, 10, 64])
    torch.Size([10, 10])

## Architecture Families

Three dominant designs reuse the same building blocks differently:

- **Encoder-only (BERT, RoBERTa)**: bidirectional self-attention over the whole input. Great for understanding tasks (classification, NER, retrieval). Pretrained with masked language modeling.
- **Decoder-only (GPT, LLaMA)**: causal self-attention; predicts the next token. The dominant design for generative LLMs. Pretrained with next-token prediction.
- **Encoder-decoder (original Transformer, T5, BART)**: encoder reads the source, decoder generates the target with cross-attention. Natural fit for translation and summarization (sequence-to-sequence).

------------------------------------------------------------------------

## Vision Transformer (ViT)

ViT applies a Transformer encoder directly to images. The image is split into fixed-size patches (e.g. 16x16); each patch is flattened and linearly projected to a token embedding. A learned `[CLS]` token is prepended and positional embeddings are added, then a standard Transformer encoder processes the sequence. The `[CLS]` output feeds a classification head.

The cleanest way to patchify-and-project in one step is a strided `Conv2d` whose kernel and stride both equal the patch size.

------------------------------------------------------------------------

``` python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch=16, in_ch=3, d_model=768):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2
        # conv with kernel=stride=patch -> one token per patch
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x):
        # x: (batch, in_ch, H, W)
        x = self.proj(x)                  # (batch, d_model, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (batch, n_patches, d_model)
        return x

pe = PatchEmbed()
img = torch.randn(2, 3, 224, 224)
tokens = pe(img)
print(tokens.shape)  # (2, 196, 768)  -> 14*14 patches
```

    torch.Size([2, 196, 768])

## Training Notes

Practical ingredients that make Transformers train well:

- **Learning-rate warmup + decay**: ramp the LR up linearly for the first few thousand steps, then decay (inverse-sqrt or cosine). Post-norm models in particular fail without warmup because early gradients are large.
- **Optimizer**: AdamW with weight decay; betas around (0.9, 0.98) for the classic schedule.
- **LayerNorm + residuals**: essential for gradient flow through deep stacks.
- **Dropout**: applied to attention weights, FFN, and embeddings for regularization.
- **Label smoothing** is common in seq2seq training.
- **Gradient clipping** (e.g. max norm 1.0) guards against rare large updates.

------------------------------------------------------------------------

``` python
# Noam-style warmup schedule from the original paper
def noam_lr(step, d_model=512, warmup=4000):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(64, 8, 256, batch_first=True), num_layers=2)
opt = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: noam_lr(s, d_model=64))
# inside the loop: opt.step(); sched.step()
print([round(noam_lr(s, 64), 5) for s in [1, 100, 4000, 8000]])
```

    [0.0, 5e-05, 0.00198, 0.0014]

## Strengths, Weaknesses, and Compute

**Strengths**

- Highly parallel training; scales to enormous models and datasets.
- Constant path length between tokens captures long-range dependencies.
- A single architecture transfers across text, vision, audio, and multimodal tasks.
- Strong transfer learning via large-scale pretraining then fine-tuning.

**Weaknesses**

- Self-attention is **O(n^2)** in time and memory with sequence length `n`, which caps practical context length. Variants (FlashAttention for speed/memory, and sparse, linear, or sliding-window attention for the quadratic term) mitigate this.
- Data-hungry: needs large pretraining corpora to beat strong CNN/RNN baselines.
- No built-in notion of locality or order; relies on positional encodings.
- Large memory footprint and inference cost; the KV-cache grows with context length.

------------------------------------------------------------------------

## Key Takeaways

- Self-attention replaces recurrence, enabling parallel training and O(1) token-to-token paths.
- Scaled dot-product attention is `softmax(Q K^T / sqrt(d_k)) V`; the scale stabilizes softmax gradients.
- Multi-head attention lets different subspaces capture different relationships.
- Positional encodings inject order because attention is permutation invariant.
- A block = attention + add&norm + FFN + add&norm; pre-norm trains deep stacks more stably than post-norm.
- Causal masking enables parallel autoregressive training in decoders.
- Families: encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5); ViT brings the same blocks to images via patch embeddings.
- The main scaling bottleneck is O(n^2) attention over context length.

------------------------------------------------------------------------

Back to top
