# RNN - Recurrent Neural Networks

Networks with loops allowing information to persist, making them suitable for sequential data. They maintain a hidden state that captures information from previous steps in the sequence.

Author

Benedict Thekkel

## Intuition

A feedforward network treats every input independently. But many problems are *sequential*: the meaning of a word depends on the words before it, and the next value of a stock depends on its history. A Recurrent Neural Network (RNN) processes a sequence one element at a time and carries a **hidden state** forward, acting as a memory of everything seen so far.

Three core ideas:

- **Sequential processing**: inputs arrive as an ordered sequence $x_1, x_2, ..., x_T$ and are consumed one timestep at a time.
- **Hidden state**: a vector $h_t$ summarizes the relevant past. At each step the network combines the current input $x_t$ with the previous state $h_{t-1}$ to produce a new state $h_t$.
- **Weight sharing across time**: the *same* weight matrices are reused at every timestep. This keeps the parameter count fixed regardless of sequence length and lets the model generalize across positions.

You can picture an RNN as a single cell with a loop: the output state is fed back as an input for the next step.

------------------------------------------------------------------------

## The Recurrence Relation

At each timestep the vanilla (Elman) RNN computes:

$$h_t = tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

and an optional output:

$$y_t = W_{hy} h_t + b_y$$

Where:

- $x_t$ is the input vector at time $t$ (size `input_size`).
- $h_t$ is the hidden state at time $t$ (size `hidden_size`); $h_0$ is usually zeros.
- $W_{xh}$ maps input to hidden, shape `(hidden_size, input_size)`.
- $W_{hh}$ maps previous hidden to hidden, shape `(hidden_size, hidden_size)`.
- $b_h$ is the hidden bias; `tanh` squashes values into `(-1, 1)`.

The same $W_{xh}, W_{hh}, b_h$ are applied at every timestep. The nonlinearity (tanh here, sometimes ReLU) is what gives the RNN expressive power over a linear recurrence.

------------------------------------------------------------------------

## From-Scratch RNN Cell (Manual Loop)

``` python
import torch

# Manual Elman RNN: loop over timesteps, reusing the same weights.
torch.manual_seed(0)

input_size, hidden_size = 4, 3
seq_len, batch = 5, 2

# Shared parameters (note the weight sharing across time)
W_xh = torch.randn(hidden_size, input_size) * 0.1   # (hidden, input)
W_hh = torch.randn(hidden_size, hidden_size) * 0.1  # (hidden, hidden)
b_h  = torch.zeros(hidden_size)                      # (hidden,)

# Input sequence: (seq_len, batch, input_size)
x = torch.randn(seq_len, batch, input_size)

h = torch.zeros(batch, hidden_size)  # h_0, shape (batch, hidden)
outputs = []
for t in range(seq_len):
    x_t = x[t]                                   # (batch, input)
    # h_t = tanh(x_t @ W_xh^T + h_{t-1} @ W_hh^T + b)
    h = torch.tanh(x_t @ W_xh.T + h @ W_hh.T + b_h)  # (batch, hidden)
    outputs.append(h)

outputs = torch.stack(outputs)  # (seq_len, batch, hidden)
print("outputs:", outputs.shape)   # torch.Size([5, 2, 3])
print("final h:", h.shape)         # torch.Size([2, 3])
```

    outputs: torch.Size([5, 2, 3])
    final h: torch.Size([2, 3])

## nn.RNN and nn.RNNCell

PyTorch provides two abstractions:

- `nn.RNNCell` is a single step: you call it once per timestep inside your own loop. Good for fine control.
- `nn.RNN` runs the whole loop internally and is much faster (optimized C++/cuDNN kernels).

Shape conventions for `nn.RNN`:

- Default input is `(seq_len, batch, input_size)`.
- With `batch_first=True` it becomes `(batch, seq_len, input_size)` which is often more intuitive.
- It returns `output` of shape `(batch, seq_len, hidden_size)` (the hidden state at *every* timestep) and `h_n` of shape `(num_layers, batch, hidden_size)` (the final hidden state per layer).

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

batch, seq_len, input_size, hidden_size = 2, 5, 4, 3

# --- nn.RNN: processes the full sequence at once ---
rnn = nn.RNN(input_size, hidden_size, batch_first=True)
x = torch.randn(batch, seq_len, input_size)     # (batch, seq, input)
output, h_n = rnn(x)
print("output:", output.shape)  # (batch, seq, hidden) = (2, 5, 3) -> h at every step
print("h_n:", h_n.shape)        # (num_layers, batch, hidden) = (1, 2, 3) -> last step

# --- nn.RNNCell: one step at a time (manual loop) ---
cell = nn.RNNCell(input_size, hidden_size)
h = torch.zeros(batch, hidden_size)             # (batch, hidden)
for t in range(seq_len):
    h = cell(x[:, t, :], h)                     # x[:, t, :] is (batch, input)
print("cell final h:", h.shape)                 # (2, 3)
```

    output: torch.Size([2, 5, 3])
    h_n: torch.Size([1, 2, 3])
    cell final h: torch.Size([2, 3])

## Unrolling Through Time and BPTT

To train an RNN we *unroll* it: conceptually we lay out one copy of the cell per timestep, forming a deep feedforward network whose layers all share the same weights. Forward pass runs left to right; the loss is computed (often summed or averaged over timesteps).

**Backpropagation Through Time (BPTT)** is just standard backprop applied to this unrolled graph. Gradients flow backward through every timestep, and because the weights are shared, the gradient contributions from all timesteps are *accumulated* into a single update for $W_{xh}, W_{hh}, b_h$.

For long sequences, unrolling the full graph is expensive and memory-hungry. **Truncated BPTT** limits how far back gradients propagate (e.g. 35 steps), trading some long-range learning for tractable cost. In PyTorch you implement it by calling `.detach()` on the hidden state between chunks so the graph does not grow without bound.

------------------------------------------------------------------------

## Vanishing and Exploding Gradients

During BPTT the gradient repeatedly multiplies by the recurrent Jacobian (roughly $W_{hh}$ scaled by the tanh derivative). Over many timesteps this product behaves like a matrix raised to a power:

- If the relevant eigenvalues are \< 1, gradients shrink toward zero: the **vanishing gradient** problem. The network cannot learn long-range dependencies because early timesteps receive almost no learning signal.
- If they are \> 1, gradients blow up: the **exploding gradient** problem, causing unstable or NaN updates.

Exploding gradients are treated with **gradient clipping**. Vanishing gradients are the deeper structural problem and motivate gated architectures (LSTM, GRU) that provide a more stable path for gradients to flow.

------------------------------------------------------------------------

``` python
import torch.nn as nn

# Gradient clipping rescales the global gradient norm if it exceeds max_norm.
# Standard remedy for exploding gradients in RNN training.
# (Sketch of a training step.)
#
# loss.backward()
# nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# optimizer.step()

# Demonstration on a tiny model:
model = nn.RNN(4, 8)
x = torch.randn(10, 2, 4)
out, _ = model(x)
out.sum().backward()
total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print("grad norm before clipping:", float(total_norm))
```

    grad norm before clipping: 67.78597259521484

## Sequence Task Shapes

RNNs map between sequences in several patterns depending on the task:

- **many-to-one**: read a whole sequence, emit one output. Example: sentiment classification of a sentence, or predicting a single label from a time series window. Use the final hidden state $h_T$.
- **one-to-many**: a single input seeds a generated sequence. Example: image captioning (one image to a sentence) or music generation from a seed.
- **many-to-many (aligned)**: one output per input timestep. Example: part-of-speech tagging, frame-level video labeling.
- **many-to-many (seq2seq)**: an encoder consumes the input sequence into a context vector, then a decoder produces an output sequence of possibly different length. Example: machine translation. This is the basis for encoder-decoder models.

------------------------------------------------------------------------

## Sine-Wave Next-Step Prediction

A classic toy regression: given a window of a sine wave, predict the next value. We use `nn.RNN` to encode the window and a linear head on the final hidden state (many-to-one).

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

class SineRNN(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, h_n = self.rnn(x)        # out: (batch, seq_len, hidden)
        last = out[:, -1, :]          # (batch, hidden) -> last timestep
        return self.fc(last)          # (batch, 1)

# Build a dataset of sliding windows over a sine wave.
seq_len = 20
t = torch.linspace(0, 8 * 3.14159, 600)
wave = torch.sin(t)
X, Y = [], []
for i in range(len(wave) - seq_len):
    X.append(wave[i:i+seq_len])
    Y.append(wave[i+seq_len])
X = torch.stack(X).unsqueeze(-1)      # (N, seq_len, 1)
Y = torch.stack(Y).unsqueeze(-1)      # (N, 1)

model = SineRNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for epoch in range(30):
    opt.zero_grad()
    pred = model(X)                   # (N, 1)
    loss = loss_fn(pred, Y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch}  loss {loss.item():.4f}")
```

    epoch 0  loss 0.5114
    epoch 10  loss 0.0442
    epoch 20  loss 0.0056

## Bidirectional and Stacked RNNs

Two common ways to add capacity:

- **Stacked (deep) RNN**: set `num_layers > 1`. The hidden sequence output by layer 1 becomes the input sequence to layer 2, and so on. This lets the model learn hierarchical temporal features. `h_n` then has shape `(num_layers, batch, hidden_size)`.
- **Bidirectional RNN**: set `bidirectional=True`. One RNN reads the sequence left-to-right, another right-to-left, and their hidden states are concatenated. Each output timestep then has context from *both* directions, which helps tasks where the whole sequence is available (tagging, classification). The output feature dimension doubles to `2 * hidden_size`. Note: bidirectional models cannot be used for online/streaming generation because they need the full sequence.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

batch, seq_len, input_size, hidden_size = 2, 5, 4, 6

rnn = nn.RNN(input_size, hidden_size,
             num_layers=3,          # stacked: 3 layers deep
             bidirectional=True,    # read both directions
             batch_first=True)

x = torch.randn(batch, seq_len, input_size)
output, h_n = rnn(x)
print("output:", output.shape)  # (batch, seq, 2*hidden) = (2, 5, 12)
print("h_n:", h_n.shape)        # (num_layers*2, batch, hidden) = (6, 2, 6)
```

    output: torch.Size([2, 5, 12])
    h_n: torch.Size([6, 2, 6])

## Strengths, Weaknesses, When to Use

**Strengths**

- Naturally handle variable-length sequences with a fixed parameter count.
- Weight sharing across time is parameter-efficient and translation-invariant in time.
- Conceptually simple and cheap per step.

**Weaknesses**

- Vanishing/exploding gradients make long-range dependencies hard to learn.
- Inherently sequential: cannot parallelize across timesteps, so training is slow on long sequences.
- Largely superseded by LSTMs/GRUs (better gradient flow) and Transformers (parallel, long-range attention) for most NLP.

**When to use**

- Short sequences, low-latency or low-resource settings, or as a lightweight baseline.
- Educational grounding before LSTM/GRU/Transformer.

------------------------------------------------------------------------

## Key Takeaways

- An RNN carries a hidden state $h_t = tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)$ across timesteps, reusing the same weights (weight sharing across time).
- `nn.RNN` runs the loop efficiently; `nn.RNNCell` is one step for manual control. Watch the input shapes and `batch_first`.
- Training uses BPTT on the unrolled graph; long sequences need truncation.
- Vanishing/exploding gradients limit long-range learning; clip gradients with `clip_grad_norm_` and reach for LSTM/GRU when memory must persist.
- Task shapes (many-to-one, one-to-many, seq2seq) determine which hidden states you read.
- `num_layers` stacks depth; `bidirectional` adds reverse-direction context and doubles the output width.

------------------------------------------------------------------------

Back to top
