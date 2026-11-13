# GAN - Generative Adversarial Networks

Composed of two neural networks, a generator and a discriminator, that are trained simultaneously through adversarial processes. The generator creates data, and the discriminator evaluates it.

Author

Benedict Thekkel

## Intuition

A **Generative Adversarial Network** pits two networks against each other in a game:

- The **Generator** $G$ takes random noise $z$ and tries to produce fake samples that look real. It is like a forger making counterfeit money.
- The **Discriminator** $D$ takes a sample and outputs the probability that it is real (from the dataset) rather than generated. It is like the police trying to detect counterfeits.

&nbsp;

    z (noise) --> [ Generator ] --> fake sample --\
                                                    >--> [ Discriminator ] --> real / fake
    real data ------------------------------------/

They train together: $D$ gets better at spotting fakes, which pushes $G$ to make better fakes. At the ideal equilibrium $G$ produces samples indistinguishable from real data, and $D$ outputs 0.5 everywhere (it can only guess).

------------------------------------------------------------------------

## The Minimax Objective

GAN training is a **two-player minimax game** with value function $V(D, G)$:

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Reading the terms:

- $D$ wants to **maximize** $V$: push $D(x) \to 1$ for real data and $D(G(z)) \to 0$ for fakes.
- $G$ wants to **minimize** $V$: push $D(G(z)) \to 1$ so its fakes fool the discriminator.

**Non-saturating loss.** In practice the $\log(1 - D(G(z)))$ term saturates (tiny gradients) early in training when fakes are obviously bad. So instead of minimizing $\log(1 - D(G(z)))$, $G$ is trained to **maximize** $\log D(G(z))$. This gives stronger gradients and is what real implementations use.

------------------------------------------------------------------------

## Generator Network

The generator maps a latent noise vector $z$ (e.g. 100-dim, drawn from $\mathcal{N}(0, I)$) to a sample. For flattened 28x28 images the output is 784-dim. `Tanh` keeps outputs in $[-1, 1]$, matching data normalized to that range.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, out_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),  nn.LeakyReLU(0.2),  # 100 -> 256
            nn.Linear(256, 512),    nn.LeakyReLU(0.2),  # 256 -> 512
            nn.Linear(512, out_dim), nn.Tanh(),         # 512 -> 784, in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)          # (B, 784)

G = Generator()
z = torch.randn(16, 100)            # (B, z_dim) noise from N(0, I)
fake = G(z)
print(fake.shape)                   # torch.Size([16, 784])
```

    torch.Size([16, 784])

## Discriminator Network

The discriminator is a binary classifier: input a sample, output the probability it is real. `Sigmoid` gives a value in $[0, 1]$. `LeakyReLU` is standard in GANs to avoid dead units.

------------------------------------------------------------------------

``` python
class Discriminator(nn.Module):
    def __init__(self, in_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LeakyReLU(0.2),  # 784 -> 512
            nn.Linear(512, 256),    nn.LeakyReLU(0.2),  # 512 -> 256
            nn.Linear(256, 1),      nn.Sigmoid(),       # 256 -> 1, prob real
        )

    def forward(self, x):
        return self.net(x)          # (B, 1)

D = Discriminator()
x = torch.rand(16, 784)
p_real = D(x)
print(p_real.shape)                 # torch.Size([16, 1])
```

    torch.Size([16, 1])

## The Training Loop

Each iteration alternates two updates:

1.  **Train D:** maximize its ability to label real as 1 and fake as 0. We detach the generator output so gradients do not flow into $G$ during the D step.
2.  **Train G:** update $G$ so $D$ labels its fakes as real (non-saturating loss: target the fakes with label 1).

We use `BCELoss` since both losses are binary cross-entropy against real/fake labels.

------------------------------------------------------------------------

``` python
z_dim = 100
G = Generator(z_dim)
D = Discriminator()
criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Synthetic stand-in for a DataLoader of flattened real images in [-1, 1]
real_loader = [torch.rand(64, 784) * 2 - 1 for _ in range(100)]

for epoch in range(3):
    for real in real_loader:
        bs = real.size(0)
        real_labels = torch.ones(bs, 1)
        fake_labels = torch.zeros(bs, 1)

        # ---- (1) Train Discriminator ----
        z = torch.randn(bs, z_dim)
        fake = G(z)
        d_real = D(real)                       # should be ~1
        d_fake = D(fake.detach())              # detach: no grad into G
        loss_D = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # ---- (2) Train Generator (non-saturating: label fakes as real) ----
        z = torch.randn(bs, z_dim)
        fake = G(z)
        d_fake = D(fake)                       # want D to output ~1 here
        loss_G = criterion(d_fake, real_labels)
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    print(f'epoch {epoch}  loss_D={loss_D.item():.3f}  loss_G={loss_G.item():.3f}')
```

    epoch 0  loss_D=1.069  loss_G=1.021
    epoch 1  loss_D=0.802  loss_G=1.204
    epoch 2  loss_D=0.332  loss_G=2.130

## Common Failure Modes and Tips

GANs are notoriously hard to train. The main failure modes:

- **Mode collapse:** $G$ produces only a few distinct outputs (e.g. one digit), ignoring the diversity of the data, because those few outputs reliably fool $D$.
- **Non-convergence:** the two losses oscillate and never settle; the adversarial game has no guaranteed convergence with gradient descent.
- **Vanishing gradients:** if $D$ becomes too strong, $D(G(z)) \to 0$ and the original $G$ loss saturates, giving $G$ almost no gradient signal.

Practical tips:

- Use the **non-saturating** generator loss (shown above).
- **Label smoothing:** use real labels of 0.9 instead of 1.0 to keep $D$ from becoming overconfident.
- **One-sided noise / label flipping** occasionally to regularize $D$.
- Adam with $\beta_1 = 0.5$, learning rate around $2 \times 10^{-4}$.
- Use **LeakyReLU** in $D$ and **BatchNorm** in $G$ (see DCGAN).
- Switch to **WGAN-GP** loss for more stable training and a meaningful loss curve.

------------------------------------------------------------------------

``` python
# Label smoothing: replace hard 1.0 targets with 0.9 for the real class.
bs = 64
real_labels = torch.full((bs, 1), 0.9)   # smoothed instead of 1.0
fake_labels = torch.zeros(bs, 1)
print(real_labels[0].item(), fake_labels[0].item())  # 0.9 0.0
```

    0.8999999761581421 0.0

## DCGAN: Convolutional GAN

**DCGAN** (Deep Convolutional GAN) replaces the fully-connected layers with convolutions, which works far better for images. Architecture guidelines:

- Generator: `ConvTranspose2d` to upsample, **BatchNorm**, **ReLU**, `Tanh` output.
- Discriminator: strided `Conv2d` to downsample, **BatchNorm**, **LeakyReLU**, `Sigmoid` output.
- No pooling layers (use strided convs for up/down sampling).
- No fully-connected hidden layers.

------------------------------------------------------------------------

``` python
class DCGenerator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, nc=1):
        super().__init__()
        # Input: (B, z_dim, 1, 1)  -> output (B, nc, 32, 32)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf*4, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),  # -> (ngf*4, 4, 4)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),  # -> (ngf*2, 8, 8)
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False), nn.BatchNorm2d(ngf),   nn.ReLU(True),  # -> (ngf, 16, 16)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh(),                                  # -> (nc, 32, 32)
        )

    def forward(self, z):
        return self.net(z)

class DCDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=1):
        super().__init__()
        # Input: (B, nc, 32, 32) -> output (B, 1, 1, 1)
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                               # -> (ndf, 16, 16)
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),     # -> (ndf*2, 8, 8)
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),   # -> (ndf*4, 4, 4)
            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False), nn.Sigmoid(),                                         # -> (1, 1, 1)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)  # (B, 1)

G = DCGenerator(); D = DCDiscriminator()
z = torch.randn(8, 100, 1, 1)
fake = G(z)
print(fake.shape, D(fake).shape)  # torch.Size([8, 1, 32, 32]) torch.Size([8, 1])
```

    torch.Size([8, 1, 32, 32]) torch.Size([8, 1])

## Conditional GAN (cGAN)

A **conditional GAN** lets us control *what* is generated by feeding a class label to both $G$ and $D$. The label is embedded into a vector and concatenated with the noise (for $G$) or the sample (for $D$). This is how you generate a *specific* digit on demand.

------------------------------------------------------------------------

``` python
class CondGenerator(nn.Module):
    def __init__(self, z_dim=100, n_classes=10, emb=10, out_dim=784):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, emb)   # label -> vector
        self.net = nn.Sequential(
            nn.Linear(z_dim + emb, 256), nn.LeakyReLU(0.2),  # noise + label
            nn.Linear(256, out_dim), nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)            # (B, emb)
        x = torch.cat([z, c], dim=1)          # (B, z_dim + emb)
        return self.net(x)                    # (B, 784)

G = CondGenerator()
z = torch.randn(4, 100)
labels = torch.tensor([0, 3, 7, 9])         # request specific classes
fake = G(z, labels)
print(fake.shape)                            # torch.Size([4, 784])
```

    torch.Size([4, 784])

## Variants Overview

- **WGAN / WGAN-GP:** replace the JS-divergence objective with the Wasserstein (earth-mover) distance. The critic outputs an unbounded score rather than a probability, which gives smoother gradients and a loss that actually correlates with sample quality. **WGAN-GP** enforces the required Lipschitz constraint with a gradient penalty instead of weight clipping. Much more stable training.
- **CycleGAN:** unpaired image-to-image translation (e.g. horses to zebras) using a cycle-consistency loss; no aligned pairs needed.
- **Pix2Pix:** *paired* image-to-image translation (e.g. edges to photo) using a conditional GAN plus an L1 reconstruction loss.
- **StyleGAN:** state-of-the-art high-resolution face generation with a style-based generator that injects per-layer style vectors, enabling fine control over attributes.

------------------------------------------------------------------------

## Evaluation Metrics

GANs have no likelihood to report, so we judge sample quality with proxy metrics:

- **Inception Score (IS):** runs generated images through a pretrained Inception classifier. High IS means each image is confidently classified (sharp, recognizable) and the set of predicted labels is diverse. **Higher is better.** It does not compare against real data, which is its main weakness.
- **Frechet Inception Distance (FID):** embeds real and generated images with Inception, fits a Gaussian to each set of features, and computes the Frechet distance between the two Gaussians. It captures both quality and diversity and *does* compare to real data. **Lower is better** and it is the most widely used GAN metric today.

------------------------------------------------------------------------

## GAN vs VAE vs Diffusion

- **GAN:** adversarial training, very **sharp** samples, fast single-pass sampling, but unstable to train and prone to mode collapse; no explicit likelihood.
- **VAE:** likelihood-based (ELBO), stable training, gives an encoder and a structured latent space, but samples are often **blurry**.
- **Diffusion models:** learn to reverse a gradual noising process. They achieve the best sample quality and diversity and train stably, but sampling is **slow** (many denoising steps, though this is improving). They have largely overtaken GANs for image generation.

------------------------------------------------------------------------

## Applications and Key Takeaways

**Applications:** photorealistic image synthesis, super-resolution, image-to-image translation, style transfer, data augmentation, inpainting, and text-to-image pipelines.

**Key takeaways:**

- A GAN is a minimax game between a generator and a discriminator.
- Train $D$ and $G$ alternately with BCE against real/fake labels; detach fakes in the D step.
- Use the non-saturating generator loss for usable gradients.
- Watch for mode collapse and non-convergence; label smoothing, careful LR, and WGAN-GP help.
- DCGAN uses strided convs, BatchNorm, LeakyReLU; conditional GANs add label embeddings.
- Evaluate with FID (lower is better) and Inception Score (higher is better).
- GANs give sharp samples but are unstable; VAEs are stable but blurry; diffusion now leads on quality at the cost of slower sampling.

------------------------------------------------------------------------

Back to top
