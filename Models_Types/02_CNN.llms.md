# CNN - Convolutional Neural Networks

Specialized neural networks for processing data with a grid-like topology, such as images. They use convolutional layers that apply filters to local regions of the input.

Author

Benedict Thekkel

## Intuition: Local Receptive Fields, Parameter Sharing, Translation Invariance

A **Convolutional Neural Network** (CNN) is built for data with a **grid topology**, most famously images. Instead of connecting every input to every neuron (as an MLP does), a CNN exploits three key ideas:

- **Local receptive fields**: each neuron looks only at a small spatial neighborhood (e.g. a 3x3 patch). Nearby pixels are correlated, so local connectivity is a strong, useful prior.
- **Parameter sharing**: the same small filter (set of weights) slides across the whole image. One 3x3x3 filter is just 27 weights regardless of image size, versus millions for a fully-connected layer. This dramatically reduces parameters and overfitting.
- **Translation invariance**: because the filter is shared across positions, a feature (an edge, a texture) is detected wherever it appears. Combined with pooling, the network becomes robust to small shifts.

Early layers learn low-level features (edges, colors); deeper layers compose them into textures, parts, and eventually whole objects.

------------------------------------------------------------------------

## The Convolution Operation (By Hand)

A 2D convolution slides a small **kernel** (filter) over the input, and at each position computes the **element-wise product summed** over the patch (technically cross-correlation, which is what deep learning frameworks actually compute). For an input patch $P$ and kernel $K$ of the same size:

$$\text{out}[i, j] = \sum_{m}\sum_{n} P[i+m, j+n] \cdot K[m, n]$$

Below is a from-scratch single-channel `conv2d` (valid padding, stride 1) in NumPy to make the mechanics concrete.

------------------------------------------------------------------------

``` python
import numpy as np

def conv2d_naive(x, k):
    # x: (H, W) input, k: (kh, kw) kernel, stride 1, no padding ('valid')
    H, W = x.shape
    kh, kw = k.shape
    out_h, out_w = H - kh + 1, W - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = x[i:i+kh, j:j+kw]      # (kh, kw)
            out[i, j] = np.sum(patch * k)  # element-wise mult then sum
    return out

img = np.array([
    [1, 2, 3, 0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 3, 0, 1],
], dtype=float)

# Vertical-edge (Sobel-like) kernel
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
], dtype=float)

feat = conv2d_naive(img, kernel)
print('input  shape:', img.shape)
print('kernel shape:', kernel.shape)
print('output shape:', feat.shape)   # (2, 2)
print(feat)
```

    input  shape: (4, 4)
    kernel shape: (3, 3)
    output shape: (2, 2)
    [[-2. -2.]
     [ 2. -2.]]

## Key Layers: Conv2d, Pooling, Padding, Stride, Dilation

PyTorch convolution layers expect input shaped **(N, C, H, W)** = (batch, channels, height, width).

- **`nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation)`**: learns `out_ch` filters, each spanning all `in_ch` input channels.
- **Padding** adds a border of zeros so output spatial size can be preserved.
- **Stride** is the step size of the sliding window; stride \> 1 downsamples.
- **Dilation** inserts gaps between kernel elements to enlarge the receptive field without extra parameters.
- **Pooling** (`nn.MaxPool2d`, `nn.AvgPool2d`) downsamples each feature map, adding translation robustness and reducing compute.

The output spatial size along each dimension is:

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

where $W$ is input size, $K$ kernel size, $P$ padding, $S$ stride (with dilation $d$, replace $K$ by $d(K-1)+1$).

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn

def out_size(W, K, P, S):
    return (W - K + 2 * P) // S + 1

x = torch.randn(1, 3, 32, 32)          # (N=1, C=3, H=32, W=32)

conv_same = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)   # keeps 32x32
conv_down = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)   # halves to 16x16
pool = nn.MaxPool2d(kernel_size=2, stride=2)                       # halves

print('conv_same out:', tuple(conv_same(x).shape),
      'formula', out_size(32, 3, 1, 1))
print('conv_down out:', tuple(conv_down(x).shape),
      'formula', out_size(32, 3, 1, 2))
print('pool      out:', tuple(pool(conv_same(x)).shape))

# Dilation enlarges the effective kernel: K_eff = d*(K-1)+1
conv_dil = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2, dilation=2)
print('dilated   out:', tuple(conv_dil(x).shape))
```

    conv_same out: (1, 16, 32, 32) formula 32
    conv_down out: (1, 16, 16, 16) formula 16
    pool      out: (1, 16, 16, 16)
    dilated   out: (1, 16, 32, 32)

## Channels and Feature Maps

An RGB image has **3 input channels**. A convolution with `out_channels = 16` produces 16 **feature maps**, each the response of one learned filter to the whole image. Each filter has shape `(in_channels, kH, kW)` and there are `out_channels` of them, so the weight tensor is `(out_channels, in_channels, kH, kW)`.

As we go deeper we typically **increase channels** (richer features) while **shrinking spatial dimensions** (via stride/pooling), trading spatial resolution for semantic depth.

------------------------------------------------------------------------

``` python
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
print('weight tensor shape:', tuple(conv.weight.shape))  # (16, 3, 3, 3)
print('bias   tensor shape:', tuple(conv.bias.shape))    # (16,)

x = torch.randn(8, 3, 28, 28)         # (N, C, H, W)
fmaps = conv(x)                       # (8, 16, 28, 28)
print('feature maps shape :', tuple(fmaps.shape))
n_params = sum(p.numel() for p in conv.parameters())
print('conv parameters    :', n_params)  # 16*3*3*3 + 16
```

    weight tensor shape: (16, 3, 3, 3)
    bias   tensor shape: (16,)
    feature maps shape : (8, 16, 28, 28)
    conv parameters    : 448

## A Small CNN Classifier (MNIST-shaped Input)

Here is a compact LeNet-style classifier for 1x28x28 grayscale digits. A common pattern is repeated **\[Conv -\> ReLU -\> Pool\]** blocks to extract features, then **flatten** and pass through fully-connected layers to produce class logits. The forward method prints intermediate shapes so you can trace how spatial size shrinks and channels grow.

------------------------------------------------------------------------

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1x28x28 -> 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32x14x14 -> 64x14x14
        self.pool = nn.MaxPool2d(2, 2)                           # halves H, W
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x, verbose=False):
        if verbose: print('input  ', tuple(x.shape))   # (N, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))           # (N, 32, 14, 14)
        if verbose: print('block1 ', tuple(x.shape))
        x = self.pool(F.relu(self.conv2(x)))           # (N, 64, 7, 7)
        if verbose: print('block2 ', tuple(x.shape))
        x = torch.flatten(x, 1)                        # (N, 64*7*7)
        if verbose: print('flatten', tuple(x.shape))
        x = F.relu(self.fc1(x))                        # (N, 128)
        return self.fc2(x)                             # (N, n_classes) logits

model = SmallCNN()
dummy = torch.randn(4, 1, 28, 28)      # batch of 4 MNIST-shaped images
logits = model(dummy, verbose=True)
print('logits ', tuple(logits.shape))  # (4, 10)
print('params ', sum(p.numel() for p in model.parameters()))
```

    input   (4, 1, 28, 28)
    block1  (4, 32, 14, 14)
    block2  (4, 64, 7, 7)
    flatten (4, 3136)
    logits  (4, 10)
    params  421642

## Receptive Field and 1x1 Convolutions

The **receptive field** of a neuron is the region of the original input that influences it. Stacking convolutions grows it: two 3x3 layers see a 5x5 region, three see 7x7. This is why deep stacks of small kernels (VGG-style) capture large context cheaply.

A **1x1 convolution** has a kernel that looks at a single spatial location but mixes across channels. It is effectively a per-pixel fully-connected layer and is used to **change the channel count** (cheaply project up or down) and add non-linearity. It is a core building block of Inception and ResNet bottleneck layers.

------------------------------------------------------------------------

``` python
x = torch.randn(2, 256, 16, 16)        # (N, C=256, H, W)

# 1x1 conv to reduce channels 256 -> 64 (a 'bottleneck'), spatial size unchanged
reduce = nn.Conv2d(256, 64, kernel_size=1)
y = reduce(x)
print('after 1x1 reduce:', tuple(y.shape))   # (2, 64, 16, 16)

# Parameter comparison: a 1x1 vs a 3x3 doing the same channel change
p_1x1 = sum(p.numel() for p in nn.Conv2d(256, 64, 1).parameters())
p_3x3 = sum(p.numel() for p in nn.Conv2d(256, 64, 3, padding=1).parameters())
print('1x1 params:', p_1x1, ' 3x3 params:', p_3x3)
```

    after 1x1 reduce: (2, 64, 16, 16)
    1x1 params: 16448  3x3 params: 147520

## Classic Architectures Overview

- **LeNet-5** (1998): the original ConvNet for digit recognition. Two conv+pool stages then fully-connected layers.
- **AlexNet** (2012): deeper, ReLU activations, dropout, GPU training; won ImageNet and launched the deep learning era.
- **VGG** (2014): very uniform design using only stacked 3x3 convolutions; simple but parameter-heavy.
- **ResNet** (2015): introduced **residual (skip) connections** that let gradients flow through identity shortcuts, enabling networks 50-150+ layers deep without degradation.
- **Inception/GoogLeNet** (2014): parallel branches with different kernel sizes (and 1x1 bottlenecks) concatenated together, capturing multi-scale features efficiently.

The residual block is the single most influential building block; here it is in PyTorch.

------------------------------------------------------------------------

``` python
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # 1x1 'projection' shortcut when shape changes, else identity
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)     # the residual / skip connection
        return F.relu(out)

block = ResidualBlock(64, 128, stride=2)
x = torch.randn(2, 64, 32, 32)
print('residual block out:', tuple(block(x).shape))  # (2, 128, 16, 16)
```

    residual block out: (2, 128, 16, 16)

## Batch Normalization in CNNs

In CNNs, **`nn.BatchNorm2d`** normalizes each channel across the batch and spatial dimensions, then applies a learnable scale and shift per channel. It stabilizes and speeds up training, allows higher learning rates, and reduces sensitivity to initialization. The conventional ordering is **Conv -\> BatchNorm -\> ReLU**, and convs before a batchnorm usually set `bias=False` because the batchnorm already provides a learnable shift.

------------------------------------------------------------------------

``` python
block = nn.Sequential(
    nn.Conv2d(16, 32, 3, padding=1, bias=False),  # bias redundant before BN
    nn.BatchNorm2d(32),                           # normalizes the 32 channels
    nn.ReLU(),
)
x = torch.randn(8, 16, 28, 28)
block.train()                       # uses batch statistics
print('train out:', tuple(block(x).shape))
block.eval()                        # uses running mean/var
print('eval  out:', tuple(block(x).shape))
```

    train out: (8, 32, 28, 28)
    eval  out: (8, 32, 28, 28)

## Transfer Learning with torchvision

Rather than training from scratch, **transfer learning** reuses a model pretrained on a large dataset (e.g. ImageNet) and adapts it to a new task. The typical recipe: load a pretrained backbone, **freeze** most of its weights, and **replace the final classification head** with one sized to your number of classes. This needs far less data and compute.

------------------------------------------------------------------------

``` python
import torch.nn as nn
from torchvision import models

# Load resnet18 pretrained on ImageNet
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all backbone parameters (feature extractor)
for p in resnet.parameters():
    p.requires_grad = False

# Replace the final fully-connected head for, say, 5 classes
num_features = resnet.fc.in_features    # 512 for resnet18
resnet.fc = nn.Linear(num_features, 5)  # new head, trainable by default

trainable = [n for n, p in resnet.named_parameters() if p.requires_grad]
print('in_features    :', num_features)
print('trainable parts:', trainable)   # only the new 'fc' layer
```

    in_features    : 512
    trainable parts: ['fc.weight', 'fc.bias']

## Data Augmentation

**Data augmentation** synthetically expands the training set by applying random, label-preserving transforms (flips, crops, color jitter, rotation). This improves generalization and is one of the cheapest, most effective regularizers for vision models. Augmentation is applied only to the **training** set; validation/test use deterministic preprocessing (resize, center crop, normalize).

------------------------------------------------------------------------

``` python
from torchvision import transforms

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # ImageNet stats
])

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
print('train transforms:', len(train_tfms.transforms))
print('eval  transforms:', len(eval_tfms.transforms))
```

    train transforms: 5
    eval  transforms: 4

## Strengths, Weaknesses, and When to Use

**Strengths**

- Parameter sharing makes them efficient and data-efficient for images.
- Built-in translation invariance and a strong spatial-locality prior.
- Hierarchical feature learning from edges to objects; pretrained backbones transfer well.

**Weaknesses**

- Limited global context per layer; long-range dependencies need deep stacks or attention.
- Not naturally suited to non-grid data (sequences, graphs, sets).
- Sensitive to large rotations/scale changes unless trained with augmentation.
- Vision Transformers can outperform CNNs given very large datasets.

**When to use**

- Images and video, spectrograms (audio as 2D), and any data with a grid structure where local patterns and translation invariance matter. For sequences and language, prefer RNNs or Transformers.

------------------------------------------------------------------------

## Key Takeaways

- CNNs replace full connectivity with **local receptive fields**, **parameter sharing**, and **translation invariance** - ideal for grid-structured data like images.
- Convolution slides a small learned kernel and computes element-wise products summed over each patch; output size follows $O = \lfloor (W - K + 2P)/S \rfloor + 1$.
- A filter spans all input channels; `out_channels` filters yield `out_channels` feature maps.
- Typical design: stacked **Conv -\> BN -\> ReLU -\> Pool** blocks, increasing channels while shrinking spatial size, then a fully-connected head.
- **1x1 convs** mix channels cheaply; **residual connections** enable very deep networks.
- **BatchNorm** stabilizes training; **augmentation** regularizes; **transfer learning** with a pretrained backbone is the practical default.
- Reach for CNNs on vision tasks; use RNNs/Transformers for sequential data.

------------------------------------------------------------------------

Back to top
