# Autoencoders

Neural networks used for unsupervised learning that aim to learn efficient codings of input data. They consist of an encoder that maps the input to a latent space and a decoder that reconstructs the input from the latent space.

Author

Benedict Thekkel

## Intuition

An **autoencoder** learns to copy its input to its output, but it is forced to do so through a narrow **bottleneck**. Because the network cannot simply pass the data through unchanged, it must discover a compact, informative representation of the data.

The architecture has two halves:

- **Encoder** $f$: maps the input $x$ to a lower-dimensional latent code $z = f(x)$.
- **Decoder** $g$: reconstructs the input from the code, $\hat{x} = g(z)$.

The model is trained to minimize the difference between $x$ and $\hat{x}$. Training is **self-supervised**: the target is the input itself, so no labels are required.

    x  -->  [ Encoder ]  -->  z (latent / bottleneck)  -->  [ Decoder ]  -->  x_hat

The latent code $z$ captures the salient structure of the data. If the bottleneck is smaller than the input (an **undercomplete** autoencoder), the network is forced to compress, which yields a useful representation for downstream tasks.

------------------------------------------------------------------------

## The Reconstruction Loss

The objective is to minimize a **reconstruction loss** measuring how far $\hat{x}$ is from $x$.

**Mean Squared Error (MSE)** is used for real-valued inputs (e.g. normalized pixels):

$$L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2$$

**Binary Cross-Entropy (BCE)** is used when inputs are in $[0, 1]$ and interpreted as probabilities (e.g. binarized MNIST). Here $\hat{x}_i$ is a sigmoid output:

$$L_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ x_i \log \hat{x}_i + (1 - x_i) \log(1 - \hat{x}_i) \right]$$

The full training objective is simply $\min_{f, g} \; \mathbb{E}_{x}[\, L(x, g(f(x))) \,]$.

------------------------------------------------------------------------

## A Basic Fully-Connected Autoencoder

We flatten a 28x28 image into a 784-dim vector, compress it down to a small latent dimension, then reconstruct. Sigmoid on the output keeps reconstructions in $[0, 1]$.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, in_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),   # 784 -> 256
            nn.Linear(256, 64),    nn.ReLU(),    # 256 -> 64
            nn.Linear(64, latent_dim),           # 64  -> 32  (bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),# 32  -> 64
            nn.Linear(64, 256),        nn.ReLU(),# 64  -> 256
            nn.Linear(256, in_dim),    nn.Sigmoid(),  # 256 -> 784, in [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)      # (B, 32)
        x_hat = self.decoder(z)  # (B, 784)
        return x_hat, z

model = Autoencoder()
x = torch.rand(16, 784)          # (B, 784) fake batch of flattened 28x28 images
x_hat, z = model(x)
print(x_hat.shape, z.shape)      # torch.Size([16, 784]) torch.Size([16, 32])
```

    torch.Size([16, 784]) torch.Size([16, 32])

A minimal training loop. In practice you would iterate over a `DataLoader` of MNIST and flatten each batch with `x.view(x.size(0), -1)`.

------------------------------------------------------------------------

``` python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Synthetic stand-in for a DataLoader: 100 batches of 64 flattened images
fake_loader = [torch.rand(64, 784) for _ in range(100)]

model.train()
for epoch in range(5):
    total = 0.0
    for x in fake_loader:               # x: (64, 784)
        x_hat, _ = model(x)             # x_hat: (64, 784)
        loss = criterion(x_hat, x)     # reconstruct the input itself
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f'epoch {epoch}  recon_loss={total/len(fake_loader):.4f}')
```

    epoch 0  recon_loss=0.0834
    epoch 1  recon_loss=0.0832
    epoch 2  recon_loss=0.0828
    epoch 3  recon_loss=0.0825
    epoch 4  recon_loss=0.0823

## Convolutional Autoencoder

For images, convolutions preserve spatial structure far better than fully-connected layers. The encoder uses strided `Conv2d` to downsample; the decoder uses `ConvTranspose2d` to upsample back to the original resolution.

------------------------------------------------------------------------

``` python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (B, 1, 28, 28)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # -> (B, 16, 14, 14)
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # -> (B, 32, 7, 7)
        )
        self.decoder = nn.Sequential(
            # ConvTranspose2d upsamples: output_padding fixes off-by-one sizes
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),  # -> (B, 16, 14, 14)
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),# -> (B, 1, 28, 28)
        )

    def forward(self, x):
        z = self.encoder(x)      # (B, 32, 7, 7)
        x_hat = self.decoder(z)  # (B, 1, 28, 28)
        return x_hat, z

model = ConvAutoencoder()
x = torch.rand(8, 1, 28, 28)
x_hat, z = model(x)
print(x_hat.shape, z.shape)  # torch.Size([8, 1, 28, 28]) torch.Size([8, 32, 7, 7])
```

    torch.Size([8, 1, 28, 28]) torch.Size([8, 32, 7, 7])

## Denoising Autoencoder

A **denoising autoencoder (DAE)** corrupts the input with noise but trains the network to reconstruct the *clean* original. This prevents the model from learning a trivial identity mapping and forces it to capture robust, meaningful structure.

Key point: the loss compares the output to the **clean** target, not the noisy input.

------------------------------------------------------------------------

``` python
def add_noise(x, noise_factor=0.3):
    noisy = x + noise_factor * torch.randn_like(x)  # add Gaussian noise
    return torch.clamp(noisy, 0.0, 1.0)             # keep valid pixel range

model = ConvAutoencoder()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

fake_loader = [torch.rand(32, 1, 28, 28) for _ in range(50)]

model.train()
for clean in fake_loader:
    noisy = add_noise(clean)            # corrupt the input
    recon, _ = model(noisy)            # feed the noisy version
    loss = criterion(recon, clean)     # but compare against the CLEAN target
    opt.zero_grad(); loss.backward(); opt.step()
print('final denoising loss:', loss.item())
```

    final denoising loss: 0.0791814997792244

## Sparse Autoencoders and the Bottleneck Idea

There are two ways to regularize an autoencoder so it learns something useful rather than the identity function:

- **Undercomplete:** the latent dimension is *smaller* than the input. Compression itself is the regularizer. This is the standard bottleneck design above.
- **Overcomplete:** the latent dimension is *equal to or larger* than the input. Without an extra constraint this can trivially copy the input, so we add a penalty.

A **sparse autoencoder** keeps a large hidden layer but penalizes activations so that only a few units fire for any given input. With an L1 penalty on the codes:

$$L = L_{recon} + \lambda \, \sum_j |z_j|$$

This encourages a sparse, disentangled representation where each unit responds to a specific feature.

------------------------------------------------------------------------

``` python
# Sparse penalty: add an L1 term on the latent activations to the reconstruction loss.
model = Autoencoder(in_dim=784, latent_dim=128)  # overcomplete-ish wide code
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
lam = 1e-3  # sparsity weight

x = torch.rand(64, 784)
x_hat, z = model(x)
recon = F.mse_loss(x_hat, x)
sparsity = z.abs().mean()          # L1 on the codes -> encourages few active units
loss = recon + lam * sparsity
opt.zero_grad(); loss.backward(); opt.step()
print(f'recon={recon.item():.4f}  sparsity={sparsity.item():.4f}')
```

    recon=0.0833  sparsity=0.0748

## Variational Autoencoder (VAE)

A plain autoencoder learns arbitrary points in latent space, so sampling new data from it is unreliable. A **Variational Autoencoder** makes the latent space *continuous and structured* by having the encoder output a **distribution** instead of a single point.

The encoder predicts a mean $\mu$ and log-variance $\log \sigma^2$ for each input. We sample $z \sim \mathcal{N}(\mu, \sigma^2)$ and decode it.

**Reparameterization trick.** Sampling is not differentiable, so we rewrite the sample as

$$z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

Now the randomness lives in $\epsilon$ (a constant w.r.t. parameters) and gradients flow through $\mu$ and $\sigma$.

**The loss (negative ELBO)** has two terms:

$$L = \underbrace{L_{recon}(x, \hat{x})}_{\text{reconstruction}} + \underbrace{D_{KL}(\, q(z|x) \,\|\, \mathcal{N}(0, I) \,)}_{\text{regularizer}}$$

For a Gaussian posterior vs. a standard normal prior the KL term has a closed form:

$$D_{KL} = -\frac{1}{2} \sum_{j} \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

The KL term pulls the latent distribution toward the prior, so we can later sample $z \sim \mathcal{N}(0, I)$ and decode to generate brand new samples.

------------------------------------------------------------------------

``` python
class VAE(nn.Module):
    def __init__(self, in_dim=784, hidden=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)      # mean
        self.fc_logvar = nn.Linear(hidden, latent_dim)  # log variance
        self.fc2 = nn.Linear(latent_dim, hidden)
        self.fc3 = nn.Linear(hidden, in_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))                 # (B, hidden)
        return self.fc_mu(h), self.fc_logvar(h) # (B, latent), (B, latent)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)           # sigma = exp(0.5 * log(sigma^2))
        eps = torch.randn_like(std)             # epsilon ~ N(0, I)
        return mu + eps * std                   # z = mu + sigma * eps

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))       # (B, 784) in [0,1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

model = VAE()
x = torch.rand(16, 784)
x_hat, mu, logvar = model(x)
print(x_hat.shape, mu.shape, logvar.shape)  # [16,784] [16,20] [16,20]
```

    torch.Size([16, 784]) torch.Size([16, 20]) torch.Size([16, 20])

``` python
def vae_loss(x_hat, x, mu, logvar):
    # Reconstruction: sum BCE over pixels, averaged over the batch.
    recon = F.binary_cross_entropy(x_hat, x, reduction='sum')
    # KL divergence between N(mu, sigma^2) and N(0, I), closed form.
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + kld) / x.size(0)  # per-sample loss

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x = torch.rand(16, 784)
x_hat, mu, logvar = model(x)
loss = vae_loss(x_hat, x, mu, logvar)
opt.zero_grad(); loss.backward(); opt.step()
print('VAE loss:', loss.item())

# Generate new samples: draw z from the prior and decode (no encoder needed).
model.eval()
with torch.no_grad():
    z = torch.randn(8, 20)            # z ~ N(0, I)
    samples = model.decode(z)         # (8, 784) brand-new generated images
print(samples.shape)
```

    VAE loss: 548.925048828125
    torch.Size([8, 784])

## Applications

- **Dimensionality reduction:** the latent code is a compact feature vector for visualization or downstream classifiers (often better than PCA for nonlinear data).
- **Anomaly detection:** train on normal data only; anomalies reconstruct poorly, so a high reconstruction error flags outliers.
- **Denoising / inpainting:** recover clean signals from corrupted inputs.
- **Generation:** VAEs (and their variants) sample new, realistic data from the prior.
- **Pretraining / representation learning:** learn features without labels, then fine-tune.

------------------------------------------------------------------------

## Autoencoder vs PCA, and AE vs GAN

**Autoencoder vs PCA.** A linear autoencoder with a single layer and MSE loss recovers the same subspace as PCA. The power of autoencoders comes from **nonlinear** activations and depth, which let them capture curved manifolds that PCA (a linear projection) cannot. PCA is cheaper, deterministic, and yields orthogonal components; autoencoders are more flexible but need training and tuning.

**AE/VAE vs GAN for generation.** VAEs optimize a likelihood-based objective (the ELBO), train stably, and give an explicit encoder, but their samples tend to be **blurry** because of the averaging effect of pixel-wise reconstruction losses. GANs use an adversarial loss and produce **sharper, more realistic** samples, but training is unstable and they provide no encoder or explicit likelihood. In short: VAEs trade sample sharpness for stability and a structured latent space.

------------------------------------------------------------------------

## Key Takeaways

- An autoencoder learns to reconstruct its input through a bottleneck, self-supervised.
- The bottleneck (undercomplete) or a penalty (sparse) forces a useful representation.
- Use MSE for continuous inputs, BCE for inputs in $[0, 1]$.
- Convolutional AEs use `Conv2d` to downsample and `ConvTranspose2d` to upsample.
- Denoising AEs reconstruct clean targets from corrupted inputs for robustness.
- VAEs make the latent space continuous via the reparameterization trick and a KL term, enabling principled generation.
- Autoencoders generalize PCA to nonlinear manifolds; VAEs trade sharpness for stable training versus GANs.

------------------------------------------------------------------------

Back to top
