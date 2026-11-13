# GRU - Gated Recurrent Unit

A simplified version of LSTM that combines the forget and input gates into a single update gate. GRUs are generally faster and require less computational resources than LSTMs.

Author

Benedict Thekkel

## Motivation

The Gated Recurrent Unit (GRU), introduced by Cho et al. (2014), is a streamlined gated RNN. It keeps the LSTM’s core benefit, gated control over memory that tames vanishing gradients, but with a simpler design:

- It uses only **two gates** (update and reset) instead of three.
- It drops the separate cell state: there is a single hidden state $h_t$ that serves as both memory and output.
- Fewer gates and one less state mean **fewer parameters** and **faster training**, often with accuracy comparable to an LSTM, especially on smaller datasets.

GRUs are a popular default when you want most of the LSTM’s power at lower computational cost.

------------------------------------------------------------------------

## The Gates and Equations

Let $\sigma$ be the sigmoid and `*` elementwise product. At each step:

**Update gate** $z_t$ - how much of the past to carry forward vs. how much new content to take (it plays the combined role of the LSTM forget and input gates):

$$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$

**Reset gate** $r_t$ - how much of the previous hidden state to use when forming the candidate (lets the unit drop irrelevant past):

$$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$$

**Candidate hidden state** - proposed new content, with the past gated by $r_t$:

$$\tilde{h}_t = tanh(W_h [r_t * h_{t-1}, x_t] + b_h)$$

**Final hidden state** - a convex (linear interpolation) blend controlled by $z_t$:

$$h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$$

When $z_t$ is near 0 the unit copies $h_{t-1}$ almost unchanged (long memory); when near 1 it overwrites with the candidate.

------------------------------------------------------------------------

## Merging Cell and Hidden State

An LSTM keeps two channels: a cell state $c_t$ (internal long-term memory) and a hidden state $h_t$ (exposed readout), with three gates coordinating them. The GRU collapses this into **one** hidden state $h_t$ that is both the memory and the output.

The single update gate $z_t$ does the job that the LSTM splits between its forget and input gates: the term $(1 - z_t) * h_{t-1}$ is the “keep” part (like forgetting selectively) and $z_t * \tilde{h}_t$ is the “write” part (like the input gate). Because the two are tied (they sum to 1), the GRU has fewer knobs but a built-in balance between remembering and updating. There is no separate output gate; the full hidden state is always exposed.

------------------------------------------------------------------------

## From-Scratch GRU Cell

``` python
import torch

torch.manual_seed(0)
input_size, hidden_size = 4, 3
batch = 2

def gru_step(x_t, h, Wz, Wr, Wh, bz, br, bh):
    # x_t: (batch, input); h: (batch, hidden)
    xh = torch.cat([h, x_t], dim=1)            # (batch, hidden+input)
    z = torch.sigmoid(xh @ Wz.T + bz)          # update gate (batch, hidden)
    r = torch.sigmoid(xh @ Wr.T + br)          # reset gate  (batch, hidden)
    xh_reset = torch.cat([r * h, x_t], dim=1)  # past gated by reset
    h_tilde = torch.tanh(xh_reset @ Wh.T + bh) # candidate   (batch, hidden)
    h = (1 - z) * h + z * h_tilde              # blended hidden state
    return h

D = hidden_size + input_size
Wz = torch.randn(hidden_size, D) * 0.1
Wr = torch.randn(hidden_size, D) * 0.1
Wh = torch.randn(hidden_size, D) * 0.1
bz = torch.zeros(hidden_size)
br = torch.zeros(hidden_size)
bh = torch.zeros(hidden_size)

x = torch.randn(5, batch, input_size)          # (seq, batch, input)
h = torch.zeros(batch, hidden_size)
for t in range(x.shape[0]):
    h = gru_step(x[t], h, Wz, Wr, Wh, bz, br, bh)
print("h:", h.shape)                           # (2, 3)
```

    h: torch.Size([2, 3])

## nn.GRU and nn.GRUCell

The GRU API matches the RNN’s exactly (single hidden state, no `(h, c)` tuple):

- `nn.GRU` processes the full sequence; returns `output` of shape `(batch, seq_len, hidden_size)` (with `batch_first=True`) and `h_n` of shape `(num_layers, batch, hidden_size)`.
- `nn.GRUCell` is a single step taking `(x_t, h)` and returning the next `h`.
- `num_layers`, `bidirectional`, and `dropout` work the same as for `nn.RNN`/`nn.LSTM`.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

batch, seq_len, input_size, hidden_size = 2, 5, 4, 3

# --- nn.GRU: full sequence ---
gru = nn.GRU(input_size, hidden_size, batch_first=True)
x = torch.randn(batch, seq_len, input_size)    # (batch, seq, input)
output, h_n = gru(x)
print("output:", output.shape)  # (batch, seq, hidden) = (2, 5, 3)
print("h_n:", h_n.shape)        # (num_layers, batch, hidden) = (1, 2, 3)

# --- nn.GRUCell: one step at a time ---
cell = nn.GRUCell(input_size, hidden_size)
h = torch.zeros(batch, hidden_size)
for t in range(seq_len):
    h = cell(x[:, t, :], h)                    # (batch, hidden)
print("cell h:", h.shape)                      # (2, 3)
```

    output: torch.Size([2, 5, 3])
    h_n: torch.Size([1, 2, 3])
    cell h: torch.Size([2, 3])

## Sequence Model and Training Loop

Same many-to-one pattern as before: encode with a GRU, read the final hidden state, apply a linear head. Minimal regression training loop below.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

class GRURegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, h_n = self.gru(x)
        last = h_n[-1]          # (batch, hidden) -> top-layer final state
        return self.fc(last)    # (batch, 1)

seq_len = 25
t = torch.linspace(0, 8 * 3.14159, 700)
wave = torch.sin(t) + 0.05 * torch.randn_like(t)
X, Y = [], []
for i in range(len(wave) - seq_len):
    X.append(wave[i:i+seq_len])
    Y.append(wave[i+seq_len])
X = torch.stack(X).unsqueeze(-1)   # (N, seq_len, 1)
Y = torch.stack(Y).unsqueeze(-1)   # (N, 1)

model = GRURegressor()
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

    epoch 0  loss 0.6782
    epoch 10  loss 0.0676
    epoch 20  loss 0.0186

## GRU vs LSTM

**Parameter count.** For hidden size $H$ and input size $I$, gated RNNs apply their gates to the concatenation of input and previous hidden ($I + H$ wide). An LSTM has 4 gate-sized transforms, a GRU has 3, so a GRU has roughly **3/4 the parameters** of an LSTM with the same dimensions. Fewer parameters means less memory and usually faster steps.

**Speed.** GRUs are generally faster to train and run (one fewer gate, no cell state to maintain).

**When each wins.**

- **GRU**: smaller datasets, tighter compute/memory budgets, when faster training matters and the task does not need the LSTM’s extra capacity. Often the better default.
- **LSTM**: very long sequences or complex tasks where the explicit, separately-gated cell state helps; large datasets where extra capacity pays off (e.g. large-scale machine translation historically favored LSTMs).

In many benchmarks the two are close; it is worth trying both. The code below counts parameters to make the ratio concrete.

------------------------------------------------------------------------

``` python
import torch.nn as nn

input_size, hidden_size = 64, 128

gru = nn.GRU(input_size, hidden_size)
lstm = nn.LSTM(input_size, hidden_size)

def count(m):
    return sum(p.numel() for p in m.parameters())

g, l = count(gru), count(lstm)
print(f"GRU params:  {g}")
print(f"LSTM params: {l}")
print(f"GRU / LSTM ratio: {g / l:.3f}")   # approx 0.75 (3 gates vs 4)
```

    GRU params:  74496
    LSTM params: 99328
    GRU / LSTM ratio: 0.750

## Key Takeaways

- A GRU is a lighter gated RNN: two gates (**update** $z_t$, **reset** $r_t$) and a **single hidden state**, no separate cell state.
- The update gate ties together remembering and writing via $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$; the reset gate decides how much past to mix into the candidate.
- `nn.GRU` has the same API as `nn.RNN` (returns `output, h_n`, no `(h, c)` tuple), and supports `num_layers`, `bidirectional`, `dropout`.
- GRUs have ~3/4 the parameters of an equivalent LSTM, train faster, and often match LSTM accuracy, especially on smaller data.
- Try both GRU and LSTM; pick GRU for speed/efficiency and LSTM when very long or complex sequences justify the extra capacity.

------------------------------------------------------------------------

Back to top
