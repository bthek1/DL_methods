# FNN - Feedforward Neural Networks

The simplest type of artificial neural network where connections between nodes do not form cycles.

Author

Benedict Thekkel

## Intuition and Where FNNs Fit

A **Feedforward Neural Network** (FNN), also called a **Multi-Layer Perceptron** (MLP) or a **fully-connected** network, is the canonical starting point of deep learning. Information flows in **one direction only**: from the input layer, through one or more hidden layers, to the output layer. There are **no cycles or feedback loops** (that is what distinguishes it from an RNN).

Each layer applies a linear transformation followed by a non-linear **activation function**. Stacking these layers lets the network learn hierarchical, non-linear feature combinations.

**Universal Approximation Theorem**: a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact domain to arbitrary precision, given a suitable activation. In practice we use **deeper** networks because depth makes representations more parameter-efficient than width.

Use an FNN when your input is a **fixed-size vector** with no strong spatial or temporal structure (tabular data, embeddings, the head of a larger model).

------------------------------------------------------------------------

## Architecture: Layers, Weights, and Biases

An FNN is organized into layers:

- **Input layer**: holds the raw feature vector $x \in \mathbb{R}^{d_{in}}$. It does no computation; it just feeds values forward.
- **Hidden layers**: each neuron computes a weighted sum of the previous layer plus a bias, then applies a non-linearity. These layers learn intermediate representations.
- **Output layer**: produces the final prediction. Its size and activation depend on the task (e.g. one logit per class for classification, a single linear unit for scalar regression).

For a layer $l$ with weight matrix $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ and bias $b^{(l)} \in \mathbb{R}^{n_l}$:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = \sigma(z^{(l)})$$

where $a^{(0)} = x$. Every neuron in layer $l$ connects to every neuron in layer $l-1$, hence **fully connected**. A layer mapping $n_{l-1}$ inputs to $n_l$ outputs has $n_l \cdot n_{l-1} + n_l$ learnable parameters.

------------------------------------------------------------------------

## The Forward Pass Math (From Scratch)

The forward pass is just repeated **affine transform + activation**: $h = \sigma(Wx + b)$. Below we implement a two-layer MLP forward pass using only NumPy so the mechanics are fully transparent. Note how matrix shapes line up: a batch of `N` examples with `d_in` features is a matrix `X` of shape `(N, d_in)`, and we right-multiply by `W.T` (or store `W` already transposed) to broadcast across the batch.

------------------------------------------------------------------------

``` python
import numpy as np

def relu(z):
    return np.maximum(0.0, z)

def softmax(z):
    # numerically stable softmax over the last axis
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

rng = np.random.default_rng(0)
N, d_in, d_hidden, d_out = 4, 3, 5, 2   # batch, in, hidden, out

X = rng.standard_normal((N, d_in))      # (4, 3)
W1 = rng.standard_normal((d_in, d_hidden)) * 0.1   # (3, 5)
b1 = np.zeros(d_hidden)                  # (5,)
W2 = rng.standard_normal((d_hidden, d_out)) * 0.1  # (5, 2)
b2 = np.zeros(d_out)                     # (2,)

z1 = X @ W1 + b1        # (4, 5) affine
a1 = relu(z1)           # (4, 5) hidden activation
z2 = a1 @ W2 + b2       # (4, 2) output logits
probs = softmax(z2)     # (4, 2) class probabilities

print('logits shape :', z2.shape)
print('probs shape  :', probs.shape)
print('rows sum to 1:', np.allclose(probs.sum(axis=1), 1.0))
```

    logits shape : (4, 2)
    probs shape  : (4, 2)
    rows sum to 1: True

## Activation Functions

Non-linear activations are what give a deep network its expressive power; without them a stack of linear layers collapses into a single linear map. Common choices:

- **ReLU** $\max(0, x)$: cheap, sparse, the default for hidden layers. Can suffer from dead neurons.
- **Sigmoid** $1/(1+e^{-x})$: squashes to $(0, 1)$. Used for binary output probabilities; saturates and causes vanishing gradients in hidden layers.
- **Tanh** $\tanh(x)$: squashes to $(-1, 1)$, zero-centered; still saturates.
- **GELU** $x \cdot \Phi(x)$: smooth, used in Transformers; often outperforms ReLU.

------------------------------------------------------------------------

``` python
import torch
import torch.nn.functional as F

x = torch.linspace(-3, 3, 7)
print('input :', x.tolist())
print('relu  :', F.relu(x).tolist())
print('sigm  :', torch.sigmoid(x).round(decimals=3).tolist())
print('tanh  :', torch.tanh(x).round(decimals=3).tolist())
print('gelu  :', F.gelu(x).round(decimals=3).tolist())

# nn.Module forms (handy inside Sequential):
import torch.nn as nn
acts = nn.ModuleList([nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.GELU()])
print([type(a).__name__ for a in acts])
```

    input : [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    relu  : [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    sigm  : [0.04699999839067459, 0.11900000274181366, 0.26899999380111694, 0.5, 0.7310000061988831, 0.8809999823570251, 0.953000009059906]
    tanh  : [-0.9950000047683716, -0.9639999866485596, -0.7620000243186951, 0.0, 0.7620000243186951, 0.9639999866485596, 0.9950000047683716]
    gelu  : [-0.004000000189989805, -0.04600000008940697, -0.1589999943971634, 0.0, 0.8410000205039978, 1.9539999961853027, 2.996000051498413]
    ['ReLU', 'Sigmoid', 'Tanh', 'GELU']

## Building an MLP in PyTorch

The idiomatic way is to subclass `nn.Module`. Define the layers in `__init__` and wire them together in `forward`. `nn.Linear(in_features, out_features)` holds the weight and bias and applies $Wx + b$.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p_drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))   # (N, d_hidden)
        x = self.drop(self.act(self.fc2(x)))   # (N, d_hidden)
        return self.fc3(x)                     # (N, d_out) raw logits

model = MLP(d_in=20, d_hidden=64, d_out=3, p_drop=0.1)
dummy = torch.randn(8, 20)             # (batch=8, features=20)
out = model(dummy)                     # (8, 3)
n_params = sum(p.numel() for p in model.parameters())
print('output shape:', tuple(out.shape))
print('parameters  :', n_params)
```

    output shape: (8, 3)
    parameters  : 5699

## The Same Network with nn.Sequential

When the data flow is a simple straight line with no branching, `nn.Sequential` is more concise. It chains modules and calls them in order.

------------------------------------------------------------------------

``` python
seq_model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 3),
)

out = seq_model(torch.randn(8, 20))    # (8, 3)
print('output shape:', tuple(out.shape))
```

    output shape: (8, 3)

## Loss Functions and When to Use Them

The loss measures how wrong the prediction is; training minimizes it.

- **`nn.CrossEntropyLoss`** for multi-class classification. It expects **raw logits** (do NOT apply softmax yourself) and integer class targets. It combines `LogSoftmax` + `NLLLoss` internally for numerical stability.
- **`nn.BCEWithLogitsLoss`** for binary or multi-label classification (logits in, 0/1 float targets).
- **`nn.MSELoss`** (mean squared error) for regression to continuous targets.
- **`nn.L1Loss`** (mean absolute error) when you want robustness to outliers.

------------------------------------------------------------------------

``` python
ce = nn.CrossEntropyLoss()
logits = torch.randn(8, 3)             # (N, C) raw, unnormalized
targets = torch.randint(0, 3, (8,))    # (N,) integer class ids
print('cross-entropy:', ce(logits, targets).item())

mse = nn.MSELoss()
pred = torch.randn(8, 1)
true = torch.randn(8, 1)
print('mse loss     :', mse(pred, true).item())
```

    cross-entropy: 0.9214856028556824
    mse loss     : 1.2620948553085327

## A Complete Minimal Training Loop

Every PyTorch training step follows the same four-beat rhythm:

1.  **Forward**: compute predictions from inputs.
2.  **Loss**: compare predictions to targets.
3.  **Backward**: `loss.backward()` computes gradients via autograd.
4.  **Step**: `optimizer.step()` updates weights; `optimizer.zero_grad()` clears old gradients.

Below we fit an MLP on a synthetic two-cluster classification problem.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

torch.manual_seed(0)

# Synthetic dataset: two Gaussian blobs -> binary-ish 2-class problem
N = 512
X0 = torch.randn(N // 2, 20) + 1.5
X1 = torch.randn(N // 2, 20) - 1.5
X = torch.cat([X0, X1], dim=0)                       # (512, 20)
y = torch.cat([torch.zeros(N // 2), torch.ones(N // 2)]).long()  # (512,)

model = MLP(d_in=20, d_hidden=64, d_out=2, p_drop=0.0)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

model.train()
for epoch in range(100):
    optimizer.zero_grad()          # 0. clear stale grads
    logits = model(X)              # 1. forward -> (512, 2)
    loss = criterion(logits, y)    # 2. loss
    loss.backward()                # 3. backward (autograd)
    optimizer.step()               # 4. update weights
    if epoch % 20 == 0:
        acc = (logits.argmax(1) == y).float().mean()
        print(f'epoch {epoch:3d}  loss {loss.item():.4f}  acc {acc.item():.3f}')
```

    epoch   0  loss 0.6855  acc 0.508
    epoch  20  loss 0.0000  acc 1.000
    epoch  40  loss 0.0000  acc 1.000
    epoch  60  loss 0.0000  acc 1.000
    epoch  80  loss 0.0000  acc 1.000

## Weight Initialization (Xavier / Kaiming)

Good initialization keeps activation and gradient variance stable across layers, which speeds up and stabilizes training.

- **Xavier/Glorot** init suits symmetric activations (tanh, sigmoid).
- **Kaiming/He** init is designed for ReLU-family activations and is the usual default.

`nn.Linear` already uses a reasonable Kaiming-uniform default, but you can set it explicitly.

------------------------------------------------------------------------

``` python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # He init for ReLU
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model = MLP(20, 64, 2)
model.apply(init_weights)   # recursively applies to every submodule
print('fc1 weight std:', model.fc1.weight.std().item())

# Xavier alternative (for tanh/sigmoid nets):
lin = nn.Linear(64, 64)
nn.init.xavier_uniform_(lin.weight)
print('xavier weight std:', lin.weight.std().item())
```

    fc1 weight std: 0.3204730451107025
    xavier weight std: 0.12445895373821259

## Regularization: Dropout, Weight Decay, BatchNorm

Regularization fights overfitting and improves generalization.

- **Dropout** randomly zeros a fraction of activations during training, preventing co-adaptation. It is disabled automatically in `model.eval()` mode.
- **Weight decay** (L2 penalty) shrinks weights; pass it to the optimizer via `weight_decay`.
- **Batch normalization** normalizes each layer’s pre-activations per mini-batch, which stabilizes and accelerates training and has a mild regularizing effect.

------------------------------------------------------------------------

``` python
reg_model = nn.Sequential(
    nn.Linear(20, 64),
    nn.BatchNorm1d(64),    # normalize 64 features across the batch
    nn.ReLU(),
    nn.Dropout(0.3),       # drop 30% of activations during training
    nn.Linear(64, 2),
)

# Weight decay (L2) is set on the optimizer, not the model:
opt = torch.optim.Adam(reg_model.parameters(), lr=1e-3, weight_decay=1e-4)

reg_model.train()                 # dropout ON, batchnorm uses batch stats
out_train = reg_model(torch.randn(8, 20))
reg_model.eval()                  # dropout OFF, batchnorm uses running stats
out_eval = reg_model(torch.randn(8, 20))
print('train out shape:', tuple(out_train.shape))
print('eval  out shape:', tuple(out_eval.shape))
```

    train out shape: (8, 2)
    eval  out shape: (8, 2)

## Strengths, Weaknesses, and When to Use

**Strengths**

- Universal approximators; flexible and simple to implement.
- Excellent for fixed-size vector inputs (tabular data, embeddings, model heads).
- Fast to train relative to convolutional or recurrent stacks of similar capacity.

**Weaknesses**

- No built-in notion of spatial or temporal structure; they ignore the geometry of images and the order of sequences.
- Fully-connected layers explode in parameter count for high-dimensional inputs (a raw 224x224x3 image is 150k inputs per neuron).
- Prone to overfitting without regularization.

**When to use which**

- **FNN/MLP**: tabular data, fixed-length feature vectors, the classifier head on top of a CNN or Transformer backbone.
- **CNN**: grid-structured data such as images, where local patterns and translation invariance matter.
- **RNN/Transformer**: sequential or variable-length data such as text, audio, or time series.

------------------------------------------------------------------------

## Key Takeaways

- An FNN is a stack of **affine + non-linear** layers with no cycles: $a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})$.
- Non-linear **activations** (ReLU by default) are essential; without them depth adds no power.
- Build them with `nn.Module` for flexibility or `nn.Sequential` for simple chains.
- Pick the loss to match the task: **CrossEntropyLoss** (classification, raw logits in) vs **MSELoss** (regression).
- The training loop is always: zero_grad, forward, loss, backward, step.
- **Kaiming** init for ReLU nets; regularize with **dropout**, **weight decay**, and **batchnorm**.
- Reach for a CNN or Transformer when the data has spatial or sequential structure an MLP cannot exploit.

------------------------------------------------------------------------

Back to top
