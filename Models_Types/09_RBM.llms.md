# RBM - Restricted Boltzmann Machines

A type of stochastic neural network that can learn a probability distribution over its set of inputs. Often used as building blocks for deep belief networks.

Author

Benedict Thekkel

## Intuition: An Energy-Based Stochastic Model

A Restricted Boltzmann Machine (RBM) is a generative, energy-based model with two layers of binary stochastic units:

- a **visible** layer `v` holding the observed data, and
- a **hidden** layer `h` of latent features.

Every visible unit connects to every hidden unit, but there are **no connections within a layer** (visible-to-visible or hidden-to-hidden). This *restriction* makes the graph bipartite and is exactly what makes inference tractable: given one layer, the units of the other layer are conditionally independent.

The model assigns an *energy* to each joint configuration `(v, h)`. Low-energy configurations are more probable. Training shapes the energy surface so that configurations resembling the training data have low energy.

------------------------------------------------------------------------

## The Energy Function and Joint Probability

For binary units the energy of a joint configuration is:

$$E(v, h) = -a^T v - b^T h - v^T W h$$

where `a` is the visible bias vector, `b` the hidden bias vector, and `W` the weight matrix coupling the two layers.

The joint probability follows the Boltzmann distribution:

$$p(v, h) = \frac{1}{Z} e^{-E(v, h)}, \quad Z = \sum_{v, h} e^{-E(v, h)}$$

`Z` is the *partition function*, a sum over all possible configurations. It is intractable for any realistic size, which is why RBM training avoids computing it directly and instead relies on sampling-based gradients.

------------------------------------------------------------------------

## Conditional Independence

Because there are no intra-layer connections, the units in each layer are conditionally independent given the other layer, and the conditionals factorize into per-unit sigmoids:

$$p(h_j = 1 \mid v) = \sigma\left(b_j + \sum_i v_i W_{ij}\right)$$

$$p(v_i = 1 \mid h) = \sigma\left(a_i + \sum_j h_j W_{ij}\right)$$

where $\sigma(x) = 1 / (1 + e^{-x})$. This is the key property: we can sample an entire layer in one parallel pass (a matrix multiply plus a sigmoid plus a Bernoulli draw). Alternating these two sampling steps is *block Gibbs sampling*.

------------------------------------------------------------------------

## Contrastive Divergence (CD-k)

The maximum-likelihood gradient of the weights has a clean form:

$$\frac{\partial \log p(v)}{\partial W_{ij}} = \langle v_i h_j \rangle_{data} - \langle v_i h_j \rangle_{model}$$

The first term (the *positive phase*) is easy: clamp `v` to a data point and compute hidden probabilities. The second term (the *negative phase*) is the model’s expectation, which would require sampling from the model to equilibrium and is intractable.

**Contrastive Divergence (CD-k)** approximates the negative phase by running only `k` steps of Gibbs sampling starting from the data, instead of running to equilibrium. `k = 1` works surprisingly well in practice. The algorithm:

1.  Start with a data vector `v0`.
2.  Sample `h0 ~ p(h | v0)` (positive phase).
3.  Sample `v1 ~ p(v | h0)` (reconstruction).
4.  Sample `h1 ~ p(h | v1)` (negative phase); repeat steps 3-4 for `k` steps.
5.  Update parameters with the difference of outer products:
    - `W += lr * (v0^T p(h|v0) - vk^T p(h|vk))`
    - `a += lr * (v0 - vk)`
    - `b += lr * (p(h|v0) - p(h|vk))`

------------------------------------------------------------------------

## A From-Scratch RBM in PyTorch

The module below stores `W`, the visible bias, and the hidden bias as parameters. It provides `sample_h` and `sample_v` (each returns the probability and a Bernoulli sample) and a `contrastive_divergence` step that applies the CD-k updates manually. We update the parameters by hand because CD is not a standard backprop loss.

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
        # v: (batch, n_visible)
        prob_h = torch.sigmoid(v @ self.W + self.h_bias)  # (batch, n_hidden)
        return prob_h, torch.bernoulli(prob_h)

    def sample_v(self, h):
        # h: (batch, n_hidden)
        prob_v = torch.sigmoid(h @ self.W.t() + self.v_bias)  # (batch, n_visible)
        return prob_v, torch.bernoulli(prob_v)

    def forward(self, v):
        # convenience: return hidden probabilities (feature representation)
        return self.sample_h(v)[0]

    @torch.no_grad()
    def contrastive_divergence(self, v0, k=1, lr=0.01):
        # positive phase
        ph0, _ = self.sample_h(v0)
        # k steps of Gibbs sampling for the negative phase
        vk = v0
        for _ in range(k):
            _, h = self.sample_h(vk)
            _, vk = self.sample_v(h)
        phk, _ = self.sample_h(vk)
        batch = v0.size(0)
        # parameter updates (averaged over the batch)
        self.W += lr * (v0.t() @ ph0 - vk.t() @ phk) / batch
        self.v_bias += lr * (v0 - vk).mean(0)
        self.h_bias += lr * (ph0 - phk).mean(0)
        return torch.mean((v0 - vk) ** 2)  # reconstruction error to monitor

rbm = RBM(n_visible=20, n_hidden=10)
print(rbm.forward(torch.bernoulli(torch.rand(4, 20))).shape)  # (4, 10)
```

    torch.Size([4, 10])

``` python
# Minimal training loop on binarized synthetic data
torch.manual_seed(0)
n_visible = 20
# create structured binary data: each sample is a noisy version of one of 3 prototypes
prototypes = torch.bernoulli(torch.rand(3, n_visible))
def make_batch(batch=64):
    idx = torch.randint(0, 3, (batch,))
    base = prototypes[idx]
    flip = torch.bernoulli(torch.full_like(base, 0.1))  # 10% noise
    return (base + flip).clamp(0, 1)  # (batch, n_visible) binary

rbm = RBM(n_visible=n_visible, n_hidden=8)
for epoch in range(100):
    v0 = make_batch()                       # (batch, n_visible)
    err = rbm.contrastive_divergence(v0, k=1, lr=0.1)
    if epoch % 20 == 0:
        print(f'epoch {epoch:3d}  recon error {err.item():.4f}')
```

    epoch   0  recon error 0.5070
    epoch  20  recon error 0.3875
    epoch  40  recon error 0.4500
    epoch  60  recon error 0.3867
    epoch  80  recon error 0.3867

## Free Energy and Feature Extraction

Marginalizing out the hidden units gives the *free energy* of a visible vector:

$$F(v) = -a^T v - \sum_j \log\left(1 + e^{b_j + (vW)_j}\right)$$

and `p(v) = e^{-F(v)} / Z`. Free energy is useful even though `Z` is unknown, because *differences* in free energy cancel `Z`. A common trick for classification: train one RBM per class (or a joint RBM over input and label), then assign a new input to the class whose free energy is lowest.

More commonly the RBM is used purely as a **feature extractor**: the hidden probabilities `p(h | v)` form a learned, lower-dimensional representation that you feed into a downstream classifier such as logistic regression.

------------------------------------------------------------------------

``` python
def free_energy(rbm, v):
    # v: (batch, n_visible) -> (batch,) free energy
    vbias_term = v @ rbm.v_bias                        # (batch,)
    wx_b = v @ rbm.W + rbm.h_bias                      # (batch, n_hidden)
    hidden_term = torch.log1p(torch.exp(wx_b)).sum(1)  # (batch,) softplus sum
    return -vbias_term - hidden_term

# use hidden activations as features for a downstream classifier
with torch.no_grad():
    features = rbm.forward(make_batch(32))  # (32, n_hidden) feature vectors
print(features.shape)
print(free_energy(rbm, make_batch(5)))      # lower = more probable under the model
```

    torch.Size([32, 8])
    tensor([ -8.5634, -11.1651, -11.4193, -11.0052, -13.4625],
           grad_fn=<SubBackward0>)

## Applications

- **Collaborative filtering**: RBMs were part of the winning ensemble for the Netflix Prize. Visible units represent user ratings; the model predicts missing ratings.
- **Feature learning / unsupervised pretraining**: hidden activations give useful features, historically used to initialize deep networks layer by layer.
- **Dimensionality reduction**: the hidden layer is a compact code, similar in spirit to an autoencoder bottleneck.
- **Topic modeling and document representation** with variants like the Replicated Softmax model.

------------------------------------------------------------------------

## Limitations

- Training relies on the CD approximation, which is biased and can be finicky to tune (learning rate, `k`, momentum, weight decay).
- The partition function `Z` is intractable, so exact likelihood and model comparison are hard.
- Sampling-based training is slower and noisier than plain backprop.
- Largely superseded: autoencoders, VAEs, GANs, and large pretrained networks now deliver better generative modeling and feature learning with simpler, end-to-end gradient training. RBMs remain valuable mostly for understanding energy-based models and the history of deep learning.

------------------------------------------------------------------------

## Key Takeaways

- An RBM is a bipartite, energy-based generative model with visible and hidden binary units and no intra-layer connections.
- Energy is `E(v,h) = -a^T v - b^T h - v^T W h`; joint probability follows the Boltzmann distribution with an intractable partition function `Z`.
- The restriction makes `p(h|v)` and `p(v|h)` factorize into sigmoids, enabling fast block Gibbs sampling.
- Contrastive Divergence (CD-k) approximates the intractable model expectation by a few Gibbs steps started from the data.
- RBMs were key to early deep learning (feature learning, collaborative filtering, stacking into DBNs) but are now largely replaced by autoencoders, VAEs, and GANs.

------------------------------------------------------------------------

Back to top
