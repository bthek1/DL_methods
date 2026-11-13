# DBN - Deep Belief Networks

Composed of multiple layers of RBMs, where each layer learns to represent the features of the input data at different levels of abstraction.

Author

Benedict Thekkel

## Intuition: A Stack of RBMs

A Deep Belief Network (DBN) is a deep generative model built by stacking Restricted Boltzmann Machines. Each RBM learns features of its input; the next RBM then learns features *of those features*, producing increasingly abstract representations as you go up the stack.

The central idea introduced by Hinton, Osindero, and Teh in 2006 is **greedy layer-wise unsupervised pretraining**: rather than trying to train a deep network all at once (which was extremely difficult at the time), train one layer at a time, freeze it, and move on. This gave deep networks a sensible initialization and helped trigger the modern deep learning revival.

------------------------------------------------------------------------

## How a DBN Is Built (Greedy Layer-Wise Pretraining)

The construction is sequential:

1.  Train `RBM1` on the raw data with contrastive divergence.
2.  Freeze `RBM1`. Pass the data through it to get hidden activations `h1 = p(h | v)`.
3.  Treat `h1` as the *visible* input and train `RBM2` on it.
4.  Freeze `RBM2`, compute `h2`, and train `RBM3` on `h2`, and so on.

Each layer is trained unsupervised, with no labels. Because every layer is just an RBM, the same CD-k machinery is reused at each level. After pretraining you have a stack of weight matrices that form a good starting point for either generation or supervised fine-tuning.

------------------------------------------------------------------------

## Generative View vs Fine-Tuning Phase

A DBN has a dual nature.

**Generative model.** The top two layers form an RBM (an undirected associative memory); the layers below form a directed *sigmoid belief network* pointing downward. To generate data you run Gibbs sampling in the top RBM, then propagate down through the directed layers. The original training can be refined generatively with the *wake-sleep* algorithm, which untangles the recognition (bottom-up) weights from the generative (top-down) weights.

**Discriminative fine-tuning.** More often the pretrained weights are simply copied into a standard feedforward MLP, a supervised output layer is added on top, and the whole network is fine-tuned end-to-end with backpropagation. Here the RBM pretraining acts as a smart weight initialization that places the network in a good basin before gradient-based training.

------------------------------------------------------------------------

## Code: RBM Building Block

We first define a compact RBM block (the same model as the RBM notebook) so we can stack several of them. It exposes `sample_h` for the forward feature pass and a `contrastive_divergence` training step.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v):
        prob = torch.sigmoid(v @ self.W + self.h_bias)  # (batch, n_hidden)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(h @ self.W.t() + self.v_bias)  # (batch, n_visible)
        return prob, torch.bernoulli(prob)

    @torch.no_grad()
    def contrastive_divergence(self, v0, k=1, lr=0.1):
        ph0, _ = self.sample_h(v0)
        vk = v0
        for _ in range(k):
            _, h = self.sample_h(vk)
            _, vk = self.sample_v(h)
        phk, _ = self.sample_h(vk)
        batch = v0.size(0)
        self.W += lr * (v0.t() @ ph0 - vk.t() @ phk) / batch
        self.v_bias += lr * (v0 - vk).mean(0)
        self.h_bias += lr * (ph0 - phk).mean(0)
        return torch.mean((v0 - vk) ** 2)
```

## Code: Stacking RBMs into a DBN

The `DBN` class holds a list of RBMs sized by a list of layer widths. `pretrain` runs greedy layer-wise CD: it trains the first RBM on the data, transforms the data through it, trains the next RBM on the transformed data, and so on.

------------------------------------------------------------------------

``` python
class DBN(nn.Module):
    def __init__(self, sizes):
        # sizes e.g. [784, 256, 128] -> two stacked RBMs
        super().__init__()
        self.rbms = nn.ModuleList(
            [RBM(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )

    @torch.no_grad()
    def transform(self, x, upto=None):
        # propagate x up through the stack using hidden probabilities
        upto = len(self.rbms) if upto is None else upto
        for rbm in self.rbms[:upto]:
            x, _ = rbm.sample_h(x)  # (batch, n_hidden) probabilities
        return x

    def pretrain(self, data_fn, epochs=50, k=1, lr=0.1):
        # data_fn() returns a fresh binary batch (batch, n_visible)
        for level, rbm in enumerate(self.rbms):
            for epoch in range(epochs):
                v = data_fn()
                v = self.transform(v, upto=level)  # outputs of frozen lower layers
                err = rbm.contrastive_divergence(v, k=k, lr=lr)
            print(f'RBM {level} pretrained, recon error {err.item():.4f}')
```

``` python
# greedy pretraining on synthetic binary data
torch.manual_seed(0)
n_visible = 64
prototypes = torch.bernoulli(torch.rand(4, n_visible))
def data_fn(batch=64):
    idx = torch.randint(0, 4, (batch,))
    base = prototypes[idx]
    flip = torch.bernoulli(torch.full_like(base, 0.1))
    return (base + flip).clamp(0, 1)  # (batch, n_visible)

dbn = DBN(sizes=[n_visible, 32, 16])
dbn.pretrain(data_fn, epochs=60, k=1, lr=0.1)
print(dbn.transform(data_fn(5)).shape)  # (5, 16) top-level features
```

    RBM 0 pretrained, recon error 0.3528
    RBM 1 pretrained, recon error 0.2655
    torch.Size([5, 16])

## Converting to an MLP for Supervised Fine-Tuning

To use the pretrained DBN for classification, copy each RBM’s `W` and hidden bias into the matching `nn.Linear` layer of a feedforward network (an RBM’s upward pass `v @ W + h_bias` is exactly a linear layer with weight `W^T` and bias `h_bias`), add a final classification layer, and fine-tune everything with ordinary backprop.

------------------------------------------------------------------------

``` python
import torch.nn.functional as F

def dbn_to_mlp(dbn, n_classes):
    layers = []
    for rbm in dbn.rbms:
        n_vis, n_hid = rbm.W.shape
        lin = nn.Linear(n_vis, n_hid)
        with torch.no_grad():
            lin.weight.copy_(rbm.W.t())   # nn.Linear computes x @ W^T + b
            lin.bias.copy_(rbm.h_bias)
        layers += [lin, nn.ReLU()]
    top_dim = dbn.rbms[-1].W.shape[1]
    layers.append(nn.Linear(top_dim, n_classes))  # fresh classification head
    return nn.Sequential(*layers)

mlp = dbn_to_mlp(dbn, n_classes=4)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
for step in range(200):
    x = data_fn()                       # (batch, n_visible)
    # synthetic labels = nearest prototype index
    y = torch.cdist(x, prototypes).argmin(1)  # (batch,)
    logits = mlp(x)                     # (batch, n_classes)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
print('final fine-tuning loss', round(loss.item(), 4))
```

    final fine-tuning loss 0.008

## Historical Importance

The 2006 DBN papers by Hinton and collaborators were a turning point. Before them, training deep networks with backprop usually failed due to poor initialization and vanishing gradients. Greedy unsupervised pretraining showed that deep models *could* be trained effectively, reigniting interest in deep learning.

The specific RBM/DBN recipe was later superseded once better activations (ReLU), normalization, careful initialization (Xavier/He), and large labeled datasets made purely supervised deep training work without pretraining. But the *idea* of unsupervised pretraining followed by fine-tuning is the direct ancestor of today’s self-supervised pretraining paradigm behind models like BERT and GPT.

------------------------------------------------------------------------

## DBN vs DBM

A **Deep Belief Network (DBN)** is a *hybrid*: the top two layers form an undirected RBM and the lower layers are a directed (top-down) belief network. Inference of the hidden states is fast and bottom-up.

A **Deep Boltzmann Machine (DBM)** is *fully undirected* at every layer. This allows both bottom-up and top-down influence during inference, which can yield better representations, but inference requires iterative mean-field updates and training is considerably more expensive. DBNs are cheaper and were far more widely used.

------------------------------------------------------------------------

## Applications and Limitations

**Applications**

- Unsupervised feature learning and pretraining for classification.
- Handwritten digit and image recognition (the classic MNIST demonstrations).
- Acoustic modeling in early deep speech recognition systems.
- Dimensionality reduction and generative modeling of binary data.

**Limitations**

- Layer-wise CD training is slow, biased, and has many hyperparameters.
- Largely obsolete for new work: end-to-end backprop with modern tricks outperforms it and is simpler.
- Binary-unit assumptions need extra machinery (e.g. Gaussian visible units) for real-valued data.

------------------------------------------------------------------------

## Key Takeaways

- A DBN is a stack of RBMs trained greedily, one layer at a time, in an unsupervised manner.
- Each layer learns features of the previous layer’s features, building a hierarchy of abstractions.
- After pretraining, the weights are either used generatively (top RBM + directed belief net, refined by wake-sleep) or copied into an MLP and fine-tuned with backprop.
- DBNs (2006, Hinton) helped revive deep learning by showing deep nets could be initialized well; their unsupervised-pretrain-then-fine-tune idea foreshadowed modern self-supervised pretraining.
- A DBN mixes directed and undirected layers; a DBM is fully undirected and more costly. Both are now mostly of historical and conceptual interest.

------------------------------------------------------------------------

Back to top
