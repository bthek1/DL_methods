# LSTM - Long Short-Term Memory Networks

A type of RNN designed to overcome the limitations of traditional RNNs, such as the vanishing gradient problem. They have special units called memory cells that can maintain information for long periods.

Author

Benedict Thekkel

## Why LSTM

A vanilla RNN struggles to carry information across many timesteps because gradients vanish (or explode) when backpropagated through the repeated tanh recurrence. In practice this means it forgets context from more than a handful of steps ago.

The Long Short-Term Memory (LSTM) network, introduced by Hochreiter and Schmidhuber (1997), fixes this with two key ideas:

- A separate **cell state** $c_t$ that runs straight through the sequence with only minor, gated linear interactions. This gives gradients an almost uninterrupted highway, largely additive rather than multiplicative.
- A set of **gates** (sigmoid-controlled valves) that decide what to forget, what to add, and what to expose. The network *learns* when to remember and when to overwrite.

The result: gradients flow much further back, so LSTMs learn dependencies spanning hundreds of timesteps.

------------------------------------------------------------------------

## The Gates and Equations

At each timestep the LSTM takes $x_t$ and the previous $(h_{t-1}, c_{t-1})$ and computes four interacting quantities. Let `[h_{t-1}, x_t]` denote concatenation, $\sigma$ the sigmoid (outputs in `(0,1)`, a soft on/off valve), and `*` elementwise product.

**Forget gate** - how much of the old cell state to keep:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Input gate** - how much of the new candidate to write:

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

**Cell candidate** - the proposed new content (tanh, in `(-1,1)`):

$$g_t = tanh(W_g [h_{t-1}, x_t] + b_g)$$

**Cell state update** - forget old, add new:

$$c_t = f_t * c_{t-1} + i_t * g_t$$

**Output gate** - how much of the cell to expose as the hidden state:

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t * tanh(c_t)$$

------------------------------------------------------------------------

## Cell State vs Hidden State

Think of the **cell state** $c_t$ as a conveyor belt running along the top of the sequence. Information rides along it with only small, gated modifications: the forget gate erases parts, the input gate adds parts. Crucially the update $c_t = f_t * c_{t-1} + i_t * g_t$ is mostly *additive*, so the gradient with respect to $c_{t-1}$ does not get repeatedly squashed the way a tanh recurrence does. This is the mechanism that defeats vanishing gradients.

The **hidden state** $h_t$ is a *filtered, exposed view* of the cell state: $h_t = o_t * tanh(c_t)$. It is what the next layer (and any output head) actually sees, and what gets fed back into the gates at the next step. So $c_t$ is long-term memory kept internal; $h_t$ is the short-term, task-facing readout.

------------------------------------------------------------------------

## From-Scratch LSTM Cell

``` python
import torch

torch.manual_seed(0)
input_size, hidden_size = 4, 3
batch = 2

# Combined weight for [h, x] -> 4 gate pre-activations, like PyTorch packs them.
# We keep separate matrices here for clarity.
def lstm_step(x_t, h, c, W, b):
    # x_t: (batch, input); h, c: (batch, hidden)
    z = torch.cat([h, x_t], dim=1) @ W.T + b   # (batch, 4*hidden)
    f, i, g, o = z.chunk(4, dim=1)             # each (batch, hidden)
    f = torch.sigmoid(f)                       # forget gate
    i = torch.sigmoid(i)                       # input gate
    g = torch.tanh(g)                          # cell candidate
    o = torch.sigmoid(o)                       # output gate
    c = f * c + i * g                          # cell state update
    h = o * torch.tanh(c)                      # hidden state
    return h, c

# Parameters: maps concat([h,x]) of size (hidden+input) to 4*hidden
W = torch.randn(4 * hidden_size, hidden_size + input_size) * 0.1
b = torch.zeros(4 * hidden_size)

x = torch.randn(5, batch, input_size)          # (seq, batch, input)
h = torch.zeros(batch, hidden_size)
c = torch.zeros(batch, hidden_size)
for t in range(x.shape[0]):
    h, c = lstm_step(x[t], h, c, W, b)
print("h:", h.shape, "c:", c.shape)            # (2, 3) (2, 3)
```

    h: torch.Size([2, 3]) c: torch.Size([2, 3])

## nn.LSTM and nn.LSTMCell

PyTorch’s LSTM mirrors the RNN API but carries a **tuple** of states `(h, c)` instead of a single hidden state.

- `nn.LSTM` processes the whole sequence; returns `output` of shape `(batch, seq_len, hidden_size)` (with `batch_first=True`) plus a tuple `(h_n, c_n)`, each `(num_layers, batch, hidden_size)`.
- `nn.LSTMCell` is a single step; you pass and receive `(h, c)`.
- If you do not provide the initial state, PyTorch defaults both `h_0` and `c_0` to zeros.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

batch, seq_len, input_size, hidden_size = 2, 5, 4, 3

# --- nn.LSTM: full sequence ---
lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
x = torch.randn(batch, seq_len, input_size)        # (batch, seq, input)
output, (h_n, c_n) = lstm(x)
print("output:", output.shape)  # (batch, seq, hidden) = (2, 5, 3)
print("h_n:", h_n.shape)        # (num_layers, batch, hidden) = (1, 2, 3)
print("c_n:", c_n.shape)        # (num_layers, batch, hidden) = (1, 2, 3)

# --- nn.LSTMCell: one step at a time ---
cell = nn.LSTMCell(input_size, hidden_size)
h = torch.zeros(batch, hidden_size)
c = torch.zeros(batch, hidden_size)
for t in range(seq_len):
    h, c = cell(x[:, t, :], (h, c))                # x[:, t, :]: (batch, input)
print("cell h:", h.shape, "cell c:", c.shape)      # (2, 3) (2, 3)
```

    output: torch.Size([2, 5, 3])
    h_n: torch.Size([1, 2, 3])
    c_n: torch.Size([1, 2, 3])
    cell h: torch.Size([2, 3]) cell c: torch.Size([2, 3])

## Stacked and Bidirectional LSTM

The same scaling knobs as the RNN apply:

- `num_layers > 1` stacks LSTM layers; the output sequence of one layer feeds the next, learning hierarchical temporal features. `(h_n, c_n)` gain a leading `num_layers` dimension.
- `bidirectional=True` runs a forward and a backward LSTM and concatenates them, doubling the output feature size to `2 * hidden_size` and the state’s first dimension to `num_layers * 2`. Use it only when the full sequence is available (not for streaming generation).
- `dropout=p` (with `num_layers > 1`) applies dropout between stacked layers for regularization.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

batch, seq_len, input_size, hidden_size = 2, 5, 4, 6

lstm = nn.LSTM(input_size, hidden_size,
               num_layers=2,
               bidirectional=True,
               dropout=0.2,
               batch_first=True)

x = torch.randn(batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)
print("output:", output.shape)  # (batch, seq, 2*hidden) = (2, 5, 12)
print("h_n:", h_n.shape)        # (num_layers*2, batch, hidden) = (4, 2, 6)
```

    output: torch.Size([2, 5, 12])
    h_n: torch.Size([4, 2, 6])

## Sequence Model and Training Loop

A typical many-to-one model: encode the sequence with an LSTM, take the final hidden state, and pass it through a linear head. Below is a tiny regression example (predict a scalar from a window), with a minimal training loop. The same skeleton becomes a sentiment classifier by swapping the loss to `CrossEntropyLoss` and emitting class logits.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

class SeqRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        last = h_n[-1]            # (batch, hidden) -> top layer final state
        return self.fc(last)      # (batch, 1)

# Toy data: predict the next value of a noisy sine window.
seq_len = 25
t = torch.linspace(0, 8 * 3.14159, 700)
wave = torch.sin(t) + 0.05 * torch.randn_like(t)
X, Y = [], []
for i in range(len(wave) - seq_len):
    X.append(wave[i:i+seq_len])
    Y.append(wave[i+seq_len])
X = torch.stack(X).unsqueeze(-1)   # (N, seq_len, 1)
Y = torch.stack(Y).unsqueeze(-1)   # (N, 1)

model = SeqRegressor()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for epoch in range(30):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch}  loss {loss.item():.4f}")
```

    epoch 0  loss 0.5512
    epoch 10  loss 0.0367
    epoch 20  loss 0.0172

## Practical Tips

- **Gradient clipping** is still useful even though LSTMs mitigate exploding gradients; `nn.utils.clip_grad_norm_(model.parameters(), max_norm)` after `backward()` keeps updates stable.
- **Packed sequences**: when a batch contains sequences of different lengths you pad them, then wrap with `pack_padded_sequence` so the LSTM skips the padding (correct states and no wasted compute), and `pad_packed_sequence` to restore a padded tensor. Sort by length or pass `enforce_sorted=False`.
- **Forget-gate bias**: initializing the forget gate bias to 1 helps the cell remember by default early in training.
- Prefer `nn.LSTM` over a manual `LSTMCell` loop for speed (cuDNN fused kernels).

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

lstm = nn.LSTM(4, 8, batch_first=True)

# Two sequences padded to length 5 but real lengths 5 and 3.
x = torch.randn(2, 5, 4)            # (batch, max_len, input)
lengths = torch.tensor([5, 3])

packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
out_packed, (h_n, c_n) = lstm(packed)
out, out_lens = pad_packed_sequence(out_packed, batch_first=True)
print("padded output:", out.shape)   # (batch, max_len, hidden) = (2, 5, 8)
print("recovered lengths:", out_lens) # tensor([5, 3])
```

    padded output: torch.Size([2, 5, 8])
    recovered lengths: tensor([5, 3])

## LSTM vs GRU vs Vanilla RNN

| Aspect | Vanilla RNN | LSTM | GRU |
|----|----|----|----|
| States carried | $h_t$ | $h_t$ and $c_t$ | $h_t$ only |
| Gates | none | forget, input, output | update, reset |
| Long-range memory | poor | excellent | very good |
| Parameters | fewest | most | moderate (~3/4 of LSTM) |
| Speed | fastest per step | slowest | faster than LSTM |
| Typical use | baselines, short seqs | long dependencies, large data | similar to LSTM, less compute |

Rule of thumb: start with a GRU or LSTM; LSTMs sometimes edge ahead on very long or complex sequences thanks to the explicit cell state, while GRUs train faster and often match LSTM quality on smaller datasets.

------------------------------------------------------------------------

## Key Takeaways

- LSTMs add a **cell state** $c_t$, a near-additive memory highway, which is why they learn long-range dependencies where vanilla RNNs fail.
- Three gates control it: **forget** ($f_t$) erases, **input** ($i_t$) writes the **candidate** ($g_t$), and **output** ($o_t$) exposes $h_t = o_t * tanh(c_t)$.
- `nn.LSTM` returns `(output, (h_n, c_n))`; the state is a `(h, c)` tuple, not a single vector.
- `num_layers`, `bidirectional`, and `dropout` scale and regularize the model just like the RNN.
- Use gradient clipping and packed sequences in real training; prefer `nn.LSTM` over manual cells for speed.

------------------------------------------------------------------------

Back to top
