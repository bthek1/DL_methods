# Resnet 18 from Scratch

Resnet 18 from Scratch

Author

Benedict Thekkel

``` python
import torch
```

``` python
torch.cuda.is_available()
```

    True

``` python
from fastbook import search_images_ddg
from fastdownload import download_url
from fastai.vision.all import *
from nbdevAuto import functions
import os
import shutil
```

``` python
import torch.nn as nn
import torch
from torch import Tensor
from typing import Type
```

``` python
class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out
```

``` python
class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 1000
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

``` python
tensor = torch.rand([1, 3, 224, 224])
model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=1000)
print(model)
```

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=1000, bias=True)
    )

``` python
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
output = model(tensor)
```

    11,689,512 total parameters.
    11,689,512 training parameters.

``` python
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org served an expired TLS certificate from some nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
def get_data(batch_size=64):
    # CIFAR10 training dataset.
    dataset_train = datasets.CIFAR10(
        root='../../data',
        train=True,
        download=True,
        transform=ToTensor(),
    )
    # CIFAR10 validation dataset.
    dataset_valid = datasets.CIFAR10(
        root='../../data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # Create data loaders.
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid, 
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, valid_loader
```

``` python
def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    "Accuracy and loss per epoch, train against validation."
    epochs = [str(e) for e in range(1, len(train_acc) + 1)]
    def curve(title, ylabel, train, valid):
        return (
            Line(init_opts=opts.InitOpts(width="100%", height="360px"))
            .add_xaxis(epochs)
            .add_yaxis(f"train {ylabel}", [round(float(v), 4) for v in train], color="#1f77b4",
                       label_opts=opts.LabelOpts(is_show=False))
            .add_yaxis(f"validation {ylabel}", [round(float(v), 4) for v in valid], color="#d62728",
                       label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title=title),
                             xaxis_opts=opts.AxisOpts(name="epoch"),
                             yaxis_opts=opts.AxisOpts(name=ylabel),
                             tooltip_opts=opts.TooltipOpts(trigger="axis"))
        )
    display(curve("Accuracy", "accuracy", train_acc, valid_acc).render_notebook())
    display(curve("Loss", "loss", train_loss, valid_loss).render_notebook())
```

``` python
import torch
from tqdm import tqdm
# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc
```

``` python
# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
```

``` python
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random

# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)
```

``` python
# Learning and training parameters.
epochs = 20
batch_size = 64
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader, valid_loader = get_data(batch_size=batch_size)
# Define model based on the argument parser string.

print('[INFO]: Training ResNet18 built from scratch...')
model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
plot_name = 'resnet_scratch'

# print(model)
# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
# Optimizer.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Loss function.
criterion = nn.CrossEntropyLoss()
```

    [INFO]: Training ResNet18 built from scratch...
    11,181,642 total parameters.
    11,181,642 training parameters.

``` python
# Lists to keep track of losses and accuracies.
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# Start the training.
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(
        model, 
        train_loader, 
        optimizer, 
        criterion,
        device
    )
    valid_epoch_loss, valid_epoch_acc = validate(
        model, 
        valid_loader, 
        criterion,
        device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)
    
# Save the loss and accuracy plots.
save_plots(
    train_acc, 
    valid_acc, 
    train_loss, 
    valid_loss, 
    name=plot_name
)
print('TRAINING COMPLETE')
```

    [INFO]: Epoch 1 of 20
    Training

    100%|██████████| 782/782 [00:07<00:00, 105.94it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 265.86it/s]

    Training loss: 1.419, training acc: 48.812
    Validation loss: 1.701, validation acc: 44.670
    --------------------------------------------------
    [INFO]: Epoch 2 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 176.37it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 269.32it/s]

    Training loss: 1.016, training acc: 63.908
    Validation loss: 1.407, validation acc: 53.070
    --------------------------------------------------
    [INFO]: Epoch 3 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 176.84it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 266.42it/s]

    Training loss: 0.828, training acc: 70.692
    Validation loss: 1.842, validation acc: 44.470
    --------------------------------------------------
    [INFO]: Epoch 4 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 176.77it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 263.48it/s]

    Training loss: 0.694, training acc: 75.504
    Validation loss: 1.175, validation acc: 61.990
    --------------------------------------------------
    [INFO]: Epoch 5 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 177.92it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 250.81it/s]

    Training loss: 0.589, training acc: 79.194
    Validation loss: 1.619, validation acc: 54.120
    --------------------------------------------------
    [INFO]: Epoch 6 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 176.41it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 260.55it/s]

    Training loss: 0.491, training acc: 82.742
    Validation loss: 2.037, validation acc: 46.450
    --------------------------------------------------
    [INFO]: Epoch 7 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 176.22it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 264.41it/s]

    Training loss: 0.403, training acc: 85.710
    Validation loss: 1.737, validation acc: 54.580
    --------------------------------------------------
    [INFO]: Epoch 8 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 174.08it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 265.68it/s]

    Training loss: 0.334, training acc: 88.046
    Validation loss: 2.078, validation acc: 51.300
    --------------------------------------------------
    [INFO]: Epoch 9 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 176.17it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 265.35it/s]

    Training loss: 0.269, training acc: 90.482
    Validation loss: 1.367, validation acc: 63.970
    --------------------------------------------------
    [INFO]: Epoch 10 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 174.25it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 271.01it/s]

    Training loss: 0.223, training acc: 92.224
    Validation loss: 1.897, validation acc: 57.430
    --------------------------------------------------
    [INFO]: Epoch 11 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 175.77it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 267.51it/s]

    Training loss: 0.182, training acc: 93.644
    Validation loss: 2.022, validation acc: 57.830
    --------------------------------------------------
    [INFO]: Epoch 12 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 175.81it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 266.68it/s]

    Training loss: 0.161, training acc: 94.324
    Validation loss: 1.820, validation acc: 60.520
    --------------------------------------------------
    [INFO]: Epoch 13 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 175.14it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 268.84it/s]

    Training loss: 0.130, training acc: 95.540
    Validation loss: 1.662, validation acc: 65.070
    --------------------------------------------------
    [INFO]: Epoch 14 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 170.91it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 253.97it/s]

    Training loss: 0.109, training acc: 96.190
    Validation loss: 1.921, validation acc: 62.490
    --------------------------------------------------
    [INFO]: Epoch 15 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 176.86it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 257.81it/s]

    Training loss: 0.100, training acc: 96.562
    Validation loss: 1.657, validation acc: 65.100
    --------------------------------------------------
    [INFO]: Epoch 16 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 172.66it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 270.57it/s]

    Training loss: 0.073, training acc: 97.406
    Validation loss: 1.620, validation acc: 68.120
    --------------------------------------------------
    [INFO]: Epoch 17 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 176.62it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 267.27it/s]

    Training loss: 0.064, training acc: 97.792
    Validation loss: 1.550, validation acc: 69.760
    --------------------------------------------------
    [INFO]: Epoch 18 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 176.83it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 263.53it/s]

    Training loss: 0.065, training acc: 97.780
    Validation loss: 3.999, validation acc: 40.650
    --------------------------------------------------
    [INFO]: Epoch 19 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 175.18it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 265.23it/s]

    Training loss: 0.082, training acc: 97.232
    Validation loss: 1.368, validation acc: 72.190
    --------------------------------------------------
    [INFO]: Epoch 20 of 20
    Training

    100%|██████████| 782/782 [00:04<00:00, 168.28it/s]

    Validation

    100%|██████████| 157/157 [00:00<00:00, 267.80it/s]

    Training loss: 0.044, training acc: 98.458
    Validation loss: 1.603, validation acc: 70.800
    --------------------------------------------------

    TRAINING COMPLETE

Back to top
