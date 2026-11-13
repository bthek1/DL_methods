# Timm - Loading Models

Timm - Loading Models

Author

Benedict Thekkel

## Using Timm

``` python
!pip list | grep timm
```

    timm                      1.0.3

``` python
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
```

``` python
timm??
```

    Type:        module
    String form: <module 'timm' from '/home/ben/miniconda3/envs/pfast/lib/python3.12/site-packages/timm/__init__.py'>
    File:        ~/miniconda3/envs/pfast/lib/python3.12/site-packages/timm/__init__.py
    Source:     
    from .version import __version__
    from .layers import is_scriptable, is_exportable, set_scriptable, set_exportable
    from .models import create_model, list_models, list_pretrained, is_model, list_modules, model_entrypoint, \
        is_model_pretrained, get_pretrained_cfg, get_pretrained_cfg_value

## List Models

``` python
print(timm.list_models("resnet*"))
```

    ['resnet10t', 'resnet14t', 'resnet18', 'resnet18d', 'resnet26', 'resnet26d', 'resnet26t', 'resnet32ts', 'resnet33ts', 'resnet34', 'resnet34d', 'resnet50', 'resnet50_gn', 'resnet50c', 'resnet50d', 'resnet50s', 'resnet50t', 'resnet51q', 'resnet61q', 'resnet101', 'resnet101c', 'resnet101d', 'resnet101s', 'resnet152', 'resnet152c', 'resnet152d', 'resnet152s', 'resnet200', 'resnet200d', 'resnetaa34d', 'resnetaa50', 'resnetaa50d', 'resnetaa101d', 'resnetblur18', 'resnetblur50', 'resnetblur50d', 'resnetblur101d', 'resnetrs50', 'resnetrs101', 'resnetrs152', 'resnetrs200', 'resnetrs270', 'resnetrs350', 'resnetrs420', 'resnetv2_50', 'resnetv2_50d', 'resnetv2_50d_evos', 'resnetv2_50d_frn', 'resnetv2_50d_gn', 'resnetv2_50t', 'resnetv2_50x1_bit', 'resnetv2_50x3_bit', 'resnetv2_101', 'resnetv2_101d', 'resnetv2_101x1_bit', 'resnetv2_101x3_bit', 'resnetv2_152', 'resnetv2_152d', 'resnetv2_152x2_bit', 'resnetv2_152x4_bit']

``` python
len(timm.list_models())
```

    1063

## Create Model

``` python
model = timm.create_model('resnet18', pretrained=True)
```

## List Pretrained

``` python
print(timm.list_pretrained("resnet*"))
```

    ['resnet10t.c3_in1k', 'resnet14t.c3_in1k', 'resnet18.a1_in1k', 'resnet18.a2_in1k', 'resnet18.a3_in1k', 'resnet18.fb_ssl_yfcc100m_ft_in1k', 'resnet18.fb_swsl_ig1b_ft_in1k', 'resnet18.gluon_in1k', 'resnet18.tv_in1k', 'resnet18d.ra2_in1k', 'resnet26.bt_in1k', 'resnet26d.bt_in1k', 'resnet26t.ra2_in1k', 'resnet32ts.ra2_in1k', 'resnet33ts.ra2_in1k', 'resnet34.a1_in1k', 'resnet34.a2_in1k', 'resnet34.a3_in1k', 'resnet34.bt_in1k', 'resnet34.gluon_in1k', 'resnet34.tv_in1k', 'resnet34d.ra2_in1k', 'resnet50.a1_in1k', 'resnet50.a1h_in1k', 'resnet50.a2_in1k', 'resnet50.a3_in1k', 'resnet50.am_in1k', 'resnet50.b1k_in1k', 'resnet50.b2k_in1k', 'resnet50.bt_in1k', 'resnet50.c1_in1k', 'resnet50.c2_in1k', 'resnet50.d_in1k', 'resnet50.fb_ssl_yfcc100m_ft_in1k', 'resnet50.fb_swsl_ig1b_ft_in1k', 'resnet50.gluon_in1k', 'resnet50.ra_in1k', 'resnet50.ram_in1k', 'resnet50.tv2_in1k', 'resnet50.tv_in1k', 'resnet50_gn.a1h_in1k', 'resnet50c.gluon_in1k', 'resnet50d.a1_in1k', 'resnet50d.a2_in1k', 'resnet50d.a3_in1k', 'resnet50d.gluon_in1k', 'resnet50d.ra2_in1k', 'resnet50s.gluon_in1k', 'resnet51q.ra2_in1k', 'resnet61q.ra2_in1k', 'resnet101.a1_in1k', 'resnet101.a1h_in1k', 'resnet101.a2_in1k', 'resnet101.a3_in1k', 'resnet101.gluon_in1k', 'resnet101.tv2_in1k', 'resnet101.tv_in1k', 'resnet101c.gluon_in1k', 'resnet101d.gluon_in1k', 'resnet101d.ra2_in1k', 'resnet101s.gluon_in1k', 'resnet152.a1_in1k', 'resnet152.a1h_in1k', 'resnet152.a2_in1k', 'resnet152.a3_in1k', 'resnet152.gluon_in1k', 'resnet152.tv2_in1k', 'resnet152.tv_in1k', 'resnet152c.gluon_in1k', 'resnet152d.gluon_in1k', 'resnet152d.ra2_in1k', 'resnet152s.gluon_in1k', 'resnet200d.ra2_in1k', 'resnetaa50.a1h_in1k', 'resnetaa50d.d_in12k', 'resnetaa50d.sw_in12k', 'resnetaa50d.sw_in12k_ft_in1k', 'resnetaa101d.sw_in12k', 'resnetaa101d.sw_in12k_ft_in1k', 'resnetblur50.bt_in1k', 'resnetrs50.tf_in1k', 'resnetrs101.tf_in1k', 'resnetrs152.tf_in1k', 'resnetrs200.tf_in1k', 'resnetrs270.tf_in1k', 'resnetrs350.tf_in1k', 'resnetrs420.tf_in1k', 'resnetv2_50.a1h_in1k', 'resnetv2_50d_evos.ah_in1k', 'resnetv2_50d_gn.ah_in1k', 'resnetv2_50x1_bit.goog_distilled_in1k', 'resnetv2_50x1_bit.goog_in21k', 'resnetv2_50x1_bit.goog_in21k_ft_in1k', 'resnetv2_50x3_bit.goog_in21k', 'resnetv2_50x3_bit.goog_in21k_ft_in1k', 'resnetv2_101.a1h_in1k', 'resnetv2_101x1_bit.goog_in21k', 'resnetv2_101x1_bit.goog_in21k_ft_in1k', 'resnetv2_101x3_bit.goog_in21k', 'resnetv2_101x3_bit.goog_in21k_ft_in1k', 'resnetv2_152x2_bit.goog_in21k', 'resnetv2_152x2_bit.goog_in21k_ft_in1k', 'resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k', 'resnetv2_152x2_bit.goog_teacher_in21k_ft_in1k_384', 'resnetv2_152x4_bit.goog_in21k', 'resnetv2_152x4_bit.goog_in21k_ft_in1k']

## Using torchvision models

``` python
import torchvision.models
```

``` python
print(torchvision.models.list_models())
```

    ['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', 'deeplabv3_resnet50', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fcn_resnet101', 'fcn_resnet50', 'fcos_resnet50_fpn', 'googlenet', 'inception_v3', 'keypointrcnn_resnet50_fpn', 'lraspp_mobilenet_v3_large', 'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'maxvit_t', 'mc3_18', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mvit_v1_b', 'mvit_v2_s', 'quantized_googlenet', 'quantized_inception_v3', 'quantized_mobilenet_v2', 'quantized_mobilenet_v3_large', 'quantized_resnet18', 'quantized_resnet50', 'quantized_resnext101_32x8d', 'quantized_resnext101_64x4d', 'quantized_shufflenet_v2_x0_5', 'quantized_shufflenet_v2_x1_0', 'quantized_shufflenet_v2_x1_5', 'quantized_shufflenet_v2_x2_0', 'r2plus1d_18', 'r3d_18', 'raft_large', 'raft_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 's3d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'swin3d_b', 'swin3d_s', 'swin3d_t', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']

## Model Summary

``` python
from torchinfo import summary
import numpy as np
```

``` python
timm.create_model?
```

``` python
model = timm.create_model('resnet50', pretrained=True, num_classes=10)
random_input = np.random.rand(1, 3, 100, 100)
```

``` python
model
```

    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
        (4): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
        (5): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act1): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (drop_block): Identity()
          (act2): ReLU(inplace=True)
          (aa): Identity()
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act3): ReLU(inplace=True)
        )
      )
      (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
      (fc): Linear(in_features=2048, out_features=10, bias=True)
    )

``` python
summary(model, input_size=random_input.shape, verbose = 0, depth  = 5, col_names = (
                "input_size",
                "output_size",
                "num_params",
                "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable",
            ))
```

    =======================================================================================================================================================================================================================
    Layer (type:depth-idx)                   Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
    =======================================================================================================================================================================================================================
    ResNet                                   [1, 3, 100, 100]          [1, 10]                   --                             --                   --                        --                        True
    ├─Conv2d: 1-1                            [1, 3, 100, 100]          [1, 64, 50, 50]           9,408                       0.04%                   [7, 7]                    23,520,000                True
    ├─BatchNorm2d: 1-2                       [1, 64, 50, 50]           [1, 64, 50, 50]           128                         0.00%                   --                        128                       True
    ├─ReLU: 1-3                              [1, 64, 50, 50]           [1, 64, 50, 50]           --                             --                   --                        --                        --
    ├─MaxPool2d: 1-4                         [1, 64, 50, 50]           [1, 64, 25, 25]           --                             --                   3                         --                        --
    ├─Sequential: 1-5                        [1, 64, 25, 25]           [1, 256, 25, 25]          --                             --                   --                        --                        True
    │    └─Bottleneck: 2-1                   [1, 64, 25, 25]           [1, 256, 25, 25]          --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-1                  [1, 64, 25, 25]           [1, 64, 25, 25]           4,096                       0.02%                   [1, 1]                    2,560,000                 True
    │    │    └─BatchNorm2d: 3-2             [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        128                       True
    │    │    └─ReLU: 3-3                    [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-4                  [1, 64, 25, 25]           [1, 64, 25, 25]           36,864                      0.16%                   [3, 3]                    23,040,000                True
    │    │    └─BatchNorm2d: 3-5             [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        128                       True
    │    │    └─Identity: 3-6                [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─ReLU: 3-7                    [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─Identity: 3-8                [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-9                  [1, 64, 25, 25]           [1, 256, 25, 25]          16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    └─BatchNorm2d: 3-10            [1, 256, 25, 25]          [1, 256, 25, 25]          512                         0.00%                   --                        512                       True
    │    │    └─Sequential: 3-11             [1, 64, 25, 25]           [1, 256, 25, 25]          --                             --                   --                        --                        True
    │    │    │    └─Conv2d: 4-1             [1, 64, 25, 25]           [1, 256, 25, 25]          16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    │    └─BatchNorm2d: 4-2        [1, 256, 25, 25]          [1, 256, 25, 25]          512                         0.00%                   --                        512                       True
    │    │    └─ReLU: 3-12                   [1, 256, 25, 25]          [1, 256, 25, 25]          --                             --                   --                        --                        --
    │    └─Bottleneck: 2-2                   [1, 256, 25, 25]          [1, 256, 25, 25]          --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-13                 [1, 256, 25, 25]          [1, 64, 25, 25]           16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    └─BatchNorm2d: 3-14            [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        128                       True
    │    │    └─ReLU: 3-15                   [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-16                 [1, 64, 25, 25]           [1, 64, 25, 25]           36,864                      0.16%                   [3, 3]                    23,040,000                True
    │    │    └─BatchNorm2d: 3-17            [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        128                       True
    │    │    └─Identity: 3-18               [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─ReLU: 3-19                   [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─Identity: 3-20               [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-21                 [1, 64, 25, 25]           [1, 256, 25, 25]          16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    └─BatchNorm2d: 3-22            [1, 256, 25, 25]          [1, 256, 25, 25]          512                         0.00%                   --                        512                       True
    │    │    └─ReLU: 3-23                   [1, 256, 25, 25]          [1, 256, 25, 25]          --                             --                   --                        --                        --
    │    └─Bottleneck: 2-3                   [1, 256, 25, 25]          [1, 256, 25, 25]          --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-24                 [1, 256, 25, 25]          [1, 64, 25, 25]           16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    └─BatchNorm2d: 3-25            [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        128                       True
    │    │    └─ReLU: 3-26                   [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-27                 [1, 64, 25, 25]           [1, 64, 25, 25]           36,864                      0.16%                   [3, 3]                    23,040,000                True
    │    │    └─BatchNorm2d: 3-28            [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        128                       True
    │    │    └─Identity: 3-29               [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─ReLU: 3-30                   [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─Identity: 3-31               [1, 64, 25, 25]           [1, 64, 25, 25]           --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-32                 [1, 64, 25, 25]           [1, 256, 25, 25]          16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    └─BatchNorm2d: 3-33            [1, 256, 25, 25]          [1, 256, 25, 25]          512                         0.00%                   --                        512                       True
    │    │    └─ReLU: 3-34                   [1, 256, 25, 25]          [1, 256, 25, 25]          --                             --                   --                        --                        --
    ├─Sequential: 1-6                        [1, 256, 25, 25]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    └─Bottleneck: 2-4                   [1, 256, 25, 25]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-35                 [1, 256, 25, 25]          [1, 128, 25, 25]          32,768                      0.14%                   [1, 1]                    20,480,000                True
    │    │    └─BatchNorm2d: 3-36            [1, 128, 25, 25]          [1, 128, 25, 25]          256                         0.00%                   --                        256                       True
    │    │    └─ReLU: 3-37                   [1, 128, 25, 25]          [1, 128, 25, 25]          --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-38                 [1, 128, 25, 25]          [1, 128, 13, 13]          147,456                     0.63%                   [3, 3]                    24,920,064                True
    │    │    └─BatchNorm2d: 3-39            [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        256                       True
    │    │    └─Identity: 3-40               [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─ReLU: 3-41                   [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Identity: 3-42               [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-43                 [1, 128, 13, 13]          [1, 512, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    └─BatchNorm2d: 3-44            [1, 512, 13, 13]          [1, 512, 13, 13]          1,024                       0.00%                   --                        1,024                     True
    │    │    └─Sequential: 3-45             [1, 256, 25, 25]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    │    └─Conv2d: 4-3             [1, 256, 25, 25]          [1, 512, 13, 13]          131,072                     0.56%                   [1, 1]                    22,151,168                True
    │    │    │    └─BatchNorm2d: 4-4        [1, 512, 13, 13]          [1, 512, 13, 13]          1,024                       0.00%                   --                        1,024                     True
    │    │    └─ReLU: 3-46                   [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        --
    │    └─Bottleneck: 2-5                   [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-47                 [1, 512, 13, 13]          [1, 128, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    └─BatchNorm2d: 3-48            [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        256                       True
    │    │    └─ReLU: 3-49                   [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-50                 [1, 128, 13, 13]          [1, 128, 13, 13]          147,456                     0.63%                   [3, 3]                    24,920,064                True
    │    │    └─BatchNorm2d: 3-51            [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        256                       True
    │    │    └─Identity: 3-52               [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─ReLU: 3-53                   [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Identity: 3-54               [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-55                 [1, 128, 13, 13]          [1, 512, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    └─BatchNorm2d: 3-56            [1, 512, 13, 13]          [1, 512, 13, 13]          1,024                       0.00%                   --                        1,024                     True
    │    │    └─ReLU: 3-57                   [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        --
    │    └─Bottleneck: 2-6                   [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-58                 [1, 512, 13, 13]          [1, 128, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    └─BatchNorm2d: 3-59            [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        256                       True
    │    │    └─ReLU: 3-60                   [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-61                 [1, 128, 13, 13]          [1, 128, 13, 13]          147,456                     0.63%                   [3, 3]                    24,920,064                True
    │    │    └─BatchNorm2d: 3-62            [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        256                       True
    │    │    └─Identity: 3-63               [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─ReLU: 3-64                   [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Identity: 3-65               [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-66                 [1, 128, 13, 13]          [1, 512, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    └─BatchNorm2d: 3-67            [1, 512, 13, 13]          [1, 512, 13, 13]          1,024                       0.00%                   --                        1,024                     True
    │    │    └─ReLU: 3-68                   [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        --
    │    └─Bottleneck: 2-7                   [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-69                 [1, 512, 13, 13]          [1, 128, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    └─BatchNorm2d: 3-70            [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        256                       True
    │    │    └─ReLU: 3-71                   [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-72                 [1, 128, 13, 13]          [1, 128, 13, 13]          147,456                     0.63%                   [3, 3]                    24,920,064                True
    │    │    └─BatchNorm2d: 3-73            [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        256                       True
    │    │    └─Identity: 3-74               [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─ReLU: 3-75                   [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Identity: 3-76               [1, 128, 13, 13]          [1, 128, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-77                 [1, 128, 13, 13]          [1, 512, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    └─BatchNorm2d: 3-78            [1, 512, 13, 13]          [1, 512, 13, 13]          1,024                       0.00%                   --                        1,024                     True
    │    │    └─ReLU: 3-79                   [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        --
    ├─Sequential: 1-7                        [1, 512, 13, 13]          [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    └─Bottleneck: 2-8                   [1, 512, 13, 13]          [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-80                 [1, 512, 13, 13]          [1, 256, 13, 13]          131,072                     0.56%                   [1, 1]                    22,151,168                True
    │    │    └─BatchNorm2d: 3-81            [1, 256, 13, 13]          [1, 256, 13, 13]          512                         0.00%                   --                        512                       True
    │    │    └─ReLU: 3-82                   [1, 256, 13, 13]          [1, 256, 13, 13]          --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-83                 [1, 256, 13, 13]          [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    └─BatchNorm2d: 3-84            [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─Identity: 3-85               [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─ReLU: 3-86                   [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Identity: 3-87               [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-88                 [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-89            [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        2,048                     True
    │    │    └─Sequential: 3-90             [1, 512, 13, 13]          [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    │    └─Conv2d: 4-5             [1, 512, 13, 13]          [1, 1024, 7, 7]           524,288                     2.23%                   [1, 1]                    25,690,112                True
    │    │    │    └─BatchNorm2d: 4-6        [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        2,048                     True
    │    │    └─ReLU: 3-91                   [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    └─Bottleneck: 2-9                   [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-92                 [1, 1024, 7, 7]           [1, 256, 7, 7]            262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-93            [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─ReLU: 3-94                   [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-95                 [1, 256, 7, 7]            [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    └─BatchNorm2d: 3-96            [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─Identity: 3-97               [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─ReLU: 3-98                   [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Identity: 3-99               [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-100                [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-101           [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        2,048                     True
    │    │    └─ReLU: 3-102                  [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    └─Bottleneck: 2-10                  [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-103                [1, 1024, 7, 7]           [1, 256, 7, 7]            262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-104           [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─ReLU: 3-105                  [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-106                [1, 256, 7, 7]            [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    └─BatchNorm2d: 3-107           [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─Identity: 3-108              [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─ReLU: 3-109                  [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Identity: 3-110              [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-111                [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-112           [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        2,048                     True
    │    │    └─ReLU: 3-113                  [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    └─Bottleneck: 2-11                  [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-114                [1, 1024, 7, 7]           [1, 256, 7, 7]            262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-115           [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─ReLU: 3-116                  [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-117                [1, 256, 7, 7]            [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    └─BatchNorm2d: 3-118           [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─Identity: 3-119              [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─ReLU: 3-120                  [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Identity: 3-121              [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-122                [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-123           [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        2,048                     True
    │    │    └─ReLU: 3-124                  [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    └─Bottleneck: 2-12                  [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-125                [1, 1024, 7, 7]           [1, 256, 7, 7]            262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-126           [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─ReLU: 3-127                  [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-128                [1, 256, 7, 7]            [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    └─BatchNorm2d: 3-129           [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─Identity: 3-130              [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─ReLU: 3-131                  [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Identity: 3-132              [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-133                [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-134           [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        2,048                     True
    │    │    └─ReLU: 3-135                  [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    └─Bottleneck: 2-13                  [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-136                [1, 1024, 7, 7]           [1, 256, 7, 7]            262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-137           [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─ReLU: 3-138                  [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-139                [1, 256, 7, 7]            [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    └─BatchNorm2d: 3-140           [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        512                       True
    │    │    └─Identity: 3-141              [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─ReLU: 3-142                  [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Identity: 3-143              [1, 256, 7, 7]            [1, 256, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-144                [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    └─BatchNorm2d: 3-145           [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        2,048                     True
    │    │    └─ReLU: 3-146                  [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    ├─Sequential: 1-8                        [1, 1024, 7, 7]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    └─Bottleneck: 2-14                  [1, 1024, 7, 7]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-147                [1, 1024, 7, 7]           [1, 512, 7, 7]            524,288                     2.23%                   [1, 1]                    25,690,112                True
    │    │    └─BatchNorm2d: 3-148           [1, 512, 7, 7]            [1, 512, 7, 7]            1,024                       0.00%                   --                        1,024                     True
    │    │    └─ReLU: 3-149                  [1, 512, 7, 7]            [1, 512, 7, 7]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-150                [1, 512, 7, 7]            [1, 512, 4, 4]            2,359,296                  10.03%                   [3, 3]                    37,748,736                True
    │    │    └─BatchNorm2d: 3-151           [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                       0.00%                   --                        1,024                     True
    │    │    └─Identity: 3-152              [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─ReLU: 3-153                  [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─Identity: 3-154              [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-155                [1, 512, 4, 4]            [1, 2048, 4, 4]           1,048,576                   4.46%                   [1, 1]                    16,777,216                True
    │    │    └─BatchNorm2d: 3-156           [1, 2048, 4, 4]           [1, 2048, 4, 4]           4,096                       0.02%                   --                        4,096                     True
    │    │    └─Sequential: 3-157            [1, 1024, 7, 7]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    │    │    └─Conv2d: 4-7             [1, 1024, 7, 7]           [1, 2048, 4, 4]           2,097,152                   8.91%                   [1, 1]                    33,554,432                True
    │    │    │    └─BatchNorm2d: 4-8        [1, 2048, 4, 4]           [1, 2048, 4, 4]           4,096                       0.02%                   --                        4,096                     True
    │    │    └─ReLU: 3-158                  [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        --
    │    └─Bottleneck: 2-15                  [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-159                [1, 2048, 4, 4]           [1, 512, 4, 4]            1,048,576                   4.46%                   [1, 1]                    16,777,216                True
    │    │    └─BatchNorm2d: 3-160           [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                       0.00%                   --                        1,024                     True
    │    │    └─ReLU: 3-161                  [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-162                [1, 512, 4, 4]            [1, 512, 4, 4]            2,359,296                  10.03%                   [3, 3]                    37,748,736                True
    │    │    └─BatchNorm2d: 3-163           [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                       0.00%                   --                        1,024                     True
    │    │    └─Identity: 3-164              [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─ReLU: 3-165                  [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─Identity: 3-166              [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-167                [1, 512, 4, 4]            [1, 2048, 4, 4]           1,048,576                   4.46%                   [1, 1]                    16,777,216                True
    │    │    └─BatchNorm2d: 3-168           [1, 2048, 4, 4]           [1, 2048, 4, 4]           4,096                       0.02%                   --                        4,096                     True
    │    │    └─ReLU: 3-169                  [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        --
    │    └─Bottleneck: 2-16                  [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    │    └─Conv2d: 3-170                [1, 2048, 4, 4]           [1, 512, 4, 4]            1,048,576                   4.46%                   [1, 1]                    16,777,216                True
    │    │    └─BatchNorm2d: 3-171           [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                       0.00%                   --                        1,024                     True
    │    │    └─ReLU: 3-172                  [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-173                [1, 512, 4, 4]            [1, 512, 4, 4]            2,359,296                  10.03%                   [3, 3]                    37,748,736                True
    │    │    └─BatchNorm2d: 3-174           [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                       0.00%                   --                        1,024                     True
    │    │    └─Identity: 3-175              [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─ReLU: 3-176                  [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─Identity: 3-177              [1, 512, 4, 4]            [1, 512, 4, 4]            --                             --                   --                        --                        --
    │    │    └─Conv2d: 3-178                [1, 512, 4, 4]            [1, 2048, 4, 4]           1,048,576                   4.46%                   [1, 1]                    16,777,216                True
    │    │    └─BatchNorm2d: 3-179           [1, 2048, 4, 4]           [1, 2048, 4, 4]           4,096                       0.02%                   --                        4,096                     True
    │    │    └─ReLU: 3-180                  [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        --
    ├─SelectAdaptivePool2d: 1-9              [1, 2048, 4, 4]           [1, 2048]                 --                             --                   --                        --                        --
    │    └─AdaptiveAvgPool2d: 2-17           [1, 2048, 4, 4]           [1, 2048, 1, 1]           --                             --                   --                        --                        --
    │    └─Flatten: 2-18                     [1, 2048, 1, 1]           [1, 2048]                 --                             --                   --                        --                        --
    ├─Linear: 1-10                           [1, 2048]                 [1, 10]                   20,490                      0.09%                   --                        20,490                    True
    =======================================================================================================================================================================================================================
    Total params: 23,528,522
    Trainable params: 23,528,522
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 995.48
    =======================================================================================================================================================================================================================
    Input size (MB): 0.12
    Forward/backward pass size (MB): 38.94
    Params size (MB): 94.11
    Estimated Total Size (MB): 133.17
    =======================================================================================================================================================================================================================

``` python
model = timm.create_model('resnetv2_50', pretrained=True, num_classes=10)
random_input = np.random.rand(1, 3, 100, 100)

summary(model, input_size=random_input.shape, verbose = 0, depth  = 5, col_names = (
                "input_size",
                "output_size",
                "num_params",
                "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable",
            ))
```

    =================================================================================================================================================================================================================================
    Layer (type:depth-idx)                             Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
    =================================================================================================================================================================================================================================
    ResNetV2                                           [1, 3, 100, 100]          [1, 10]                   --                             --                   --                        --                        True
    ├─Sequential: 1-1                                  [1, 3, 100, 100]          [1, 64, 25, 25]           --                             --                   --                        --                        True
    │    └─Conv2d: 2-1                                 [1, 3, 100, 100]          [1, 64, 50, 50]           9,408                       0.04%                   [7, 7]                    23,520,000                True
    │    └─MaxPool2d: 2-2                              [1, 64, 50, 50]           [1, 64, 25, 25]           --                             --                   3                         --                        --
    ├─Sequential: 1-2                                  [1, 64, 25, 25]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    └─ResNetStage: 2-3                            [1, 64, 25, 25]           [1, 256, 25, 25]          --                             --                   --                        --                        True
    │    │    └─Sequential: 3-1                        [1, 64, 25, 25]           [1, 256, 25, 25]          --                             --                   --                        --                        True
    │    │    │    └─PreActBottleneck: 4-1             [1, 64, 25, 25]           [1, 256, 25, 25]          --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-1          [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        0                         True
    │    │    │    │    └─DownsampleConv: 5-2          [1, 64, 25, 25]           [1, 256, 25, 25]          16,384                      0.07%                   --                        10,240,000                True
    │    │    │    │    └─Conv2d: 5-3                  [1, 64, 25, 25]           [1, 64, 25, 25]           4,096                       0.02%                   [1, 1]                    2,560,000                 True
    │    │    │    │    └─BatchNormAct2d: 5-4          [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-5                  [1, 64, 25, 25]           [1, 64, 25, 25]           36,864                      0.16%                   [3, 3]                    23,040,000                True
    │    │    │    │    └─BatchNormAct2d: 5-6          [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-7                  [1, 64, 25, 25]           [1, 256, 25, 25]          16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    │    │    └─Identity: 5-8                [1, 256, 25, 25]          [1, 256, 25, 25]          --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-2             [1, 256, 25, 25]          [1, 256, 25, 25]          --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-9          [1, 256, 25, 25]          [1, 256, 25, 25]          512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-10                 [1, 256, 25, 25]          [1, 64, 25, 25]           16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    │    │    └─BatchNormAct2d: 5-11         [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-12                 [1, 64, 25, 25]           [1, 64, 25, 25]           36,864                      0.16%                   [3, 3]                    23,040,000                True
    │    │    │    │    └─BatchNormAct2d: 5-13         [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-14                 [1, 64, 25, 25]           [1, 256, 25, 25]          16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    │    │    └─Identity: 5-15               [1, 256, 25, 25]          [1, 256, 25, 25]          --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-3             [1, 256, 25, 25]          [1, 256, 25, 25]          --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-16         [1, 256, 25, 25]          [1, 256, 25, 25]          512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-17                 [1, 256, 25, 25]          [1, 64, 25, 25]           16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    │    │    └─BatchNormAct2d: 5-18         [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-19                 [1, 64, 25, 25]           [1, 64, 25, 25]           36,864                      0.16%                   [3, 3]                    23,040,000                True
    │    │    │    │    └─BatchNormAct2d: 5-20         [1, 64, 25, 25]           [1, 64, 25, 25]           128                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-21                 [1, 64, 25, 25]           [1, 256, 25, 25]          16,384                      0.07%                   [1, 1]                    10,240,000                True
    │    │    │    │    └─Identity: 5-22               [1, 256, 25, 25]          [1, 256, 25, 25]          --                             --                   --                        --                        --
    │    └─ResNetStage: 2-4                            [1, 256, 25, 25]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    └─Sequential: 3-2                        [1, 256, 25, 25]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    │    └─PreActBottleneck: 4-4             [1, 256, 25, 25]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-23         [1, 256, 25, 25]          [1, 256, 25, 25]          512                         0.00%                   --                        0                         True
    │    │    │    │    └─DownsampleConv: 5-24         [1, 256, 25, 25]          [1, 512, 13, 13]          131,072                     0.56%                   --                        22,151,168                True
    │    │    │    │    └─Conv2d: 5-25                 [1, 256, 25, 25]          [1, 128, 25, 25]          32,768                      0.14%                   [1, 1]                    20,480,000                True
    │    │    │    │    └─BatchNormAct2d: 5-26         [1, 128, 25, 25]          [1, 128, 25, 25]          256                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-27                 [1, 128, 25, 25]          [1, 128, 13, 13]          147,456                     0.63%                   [3, 3]                    24,920,064                True
    │    │    │    │    └─BatchNormAct2d: 5-28         [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-29                 [1, 128, 13, 13]          [1, 512, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    │    │    └─Identity: 5-30               [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-5             [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-31         [1, 512, 13, 13]          [1, 512, 13, 13]          1,024                       0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-32                 [1, 512, 13, 13]          [1, 128, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    │    │    └─BatchNormAct2d: 5-33         [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-34                 [1, 128, 13, 13]          [1, 128, 13, 13]          147,456                     0.63%                   [3, 3]                    24,920,064                True
    │    │    │    │    └─BatchNormAct2d: 5-35         [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-36                 [1, 128, 13, 13]          [1, 512, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    │    │    └─Identity: 5-37               [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-6             [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-38         [1, 512, 13, 13]          [1, 512, 13, 13]          1,024                       0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-39                 [1, 512, 13, 13]          [1, 128, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    │    │    └─BatchNormAct2d: 5-40         [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-41                 [1, 128, 13, 13]          [1, 128, 13, 13]          147,456                     0.63%                   [3, 3]                    24,920,064                True
    │    │    │    │    └─BatchNormAct2d: 5-42         [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-43                 [1, 128, 13, 13]          [1, 512, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    │    │    └─Identity: 5-44               [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-7             [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-45         [1, 512, 13, 13]          [1, 512, 13, 13]          1,024                       0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-46                 [1, 512, 13, 13]          [1, 128, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    │    │    └─BatchNormAct2d: 5-47         [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-48                 [1, 128, 13, 13]          [1, 128, 13, 13]          147,456                     0.63%                   [3, 3]                    24,920,064                True
    │    │    │    │    └─BatchNormAct2d: 5-49         [1, 128, 13, 13]          [1, 128, 13, 13]          256                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-50                 [1, 128, 13, 13]          [1, 512, 13, 13]          65,536                      0.28%                   [1, 1]                    11,075,584                True
    │    │    │    │    └─Identity: 5-51               [1, 512, 13, 13]          [1, 512, 13, 13]          --                             --                   --                        --                        --
    │    └─ResNetStage: 2-5                            [1, 512, 13, 13]          [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    └─Sequential: 3-3                        [1, 512, 13, 13]          [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    │    └─PreActBottleneck: 4-8             [1, 512, 13, 13]          [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-52         [1, 512, 13, 13]          [1, 512, 13, 13]          1,024                       0.00%                   --                        0                         True
    │    │    │    │    └─DownsampleConv: 5-53         [1, 512, 13, 13]          [1, 1024, 7, 7]           524,288                     2.23%                   --                        25,690,112                True
    │    │    │    │    └─Conv2d: 5-54                 [1, 512, 13, 13]          [1, 256, 13, 13]          131,072                     0.56%                   [1, 1]                    22,151,168                True
    │    │    │    │    └─BatchNormAct2d: 5-55         [1, 256, 13, 13]          [1, 256, 13, 13]          512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-56                 [1, 256, 13, 13]          [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    │    │    └─BatchNormAct2d: 5-57         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-58                 [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─Identity: 5-59               [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-9             [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-60         [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-61                 [1, 1024, 7, 7]           [1, 256, 7, 7]            262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─BatchNormAct2d: 5-62         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-63                 [1, 256, 7, 7]            [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    │    │    └─BatchNormAct2d: 5-64         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-65                 [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─Identity: 5-66               [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-10            [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-67         [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-68                 [1, 1024, 7, 7]           [1, 256, 7, 7]            262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─BatchNormAct2d: 5-69         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-70                 [1, 256, 7, 7]            [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    │    │    └─BatchNormAct2d: 5-71         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-72                 [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─Identity: 5-73               [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-11            [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-74         [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-75                 [1, 1024, 7, 7]           [1, 256, 7, 7]            262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─BatchNormAct2d: 5-76         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-77                 [1, 256, 7, 7]            [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    │    │    └─BatchNormAct2d: 5-78         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-79                 [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─Identity: 5-80               [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-12            [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-81         [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-82                 [1, 1024, 7, 7]           [1, 256, 7, 7]            262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─BatchNormAct2d: 5-83         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-84                 [1, 256, 7, 7]            [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    │    │    └─BatchNormAct2d: 5-85         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-86                 [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─Identity: 5-87               [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-13            [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-88         [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-89                 [1, 1024, 7, 7]           [1, 256, 7, 7]            262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─BatchNormAct2d: 5-90         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-91                 [1, 256, 7, 7]            [1, 256, 7, 7]            589,824                     2.51%                   [3, 3]                    28,901,376                True
    │    │    │    │    └─BatchNormAct2d: 5-92         [1, 256, 7, 7]            [1, 256, 7, 7]            512                         0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-93                 [1, 256, 7, 7]            [1, 1024, 7, 7]           262,144                     1.11%                   [1, 1]                    12,845,056                True
    │    │    │    │    └─Identity: 5-94               [1, 1024, 7, 7]           [1, 1024, 7, 7]           --                             --                   --                        --                        --
    │    └─ResNetStage: 2-6                            [1, 1024, 7, 7]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    │    └─Sequential: 3-4                        [1, 1024, 7, 7]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    │    │    └─PreActBottleneck: 4-14            [1, 1024, 7, 7]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-95         [1, 1024, 7, 7]           [1, 1024, 7, 7]           2,048                       0.01%                   --                        0                         True
    │    │    │    │    └─DownsampleConv: 5-96         [1, 1024, 7, 7]           [1, 2048, 4, 4]           2,097,152                   8.92%                   --                        33,554,432                True
    │    │    │    │    └─Conv2d: 5-97                 [1, 1024, 7, 7]           [1, 512, 7, 7]            524,288                     2.23%                   [1, 1]                    25,690,112                True
    │    │    │    │    └─BatchNormAct2d: 5-98         [1, 512, 7, 7]            [1, 512, 7, 7]            1,024                       0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-99                 [1, 512, 7, 7]            [1, 512, 4, 4]            2,359,296                  10.03%                   [3, 3]                    37,748,736                True
    │    │    │    │    └─BatchNormAct2d: 5-100        [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                       0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-101                [1, 512, 4, 4]            [1, 2048, 4, 4]           1,048,576                   4.46%                   [1, 1]                    16,777,216                True
    │    │    │    │    └─Identity: 5-102              [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-15            [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-103        [1, 2048, 4, 4]           [1, 2048, 4, 4]           4,096                       0.02%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-104                [1, 2048, 4, 4]           [1, 512, 4, 4]            1,048,576                   4.46%                   [1, 1]                    16,777,216                True
    │    │    │    │    └─BatchNormAct2d: 5-105        [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                       0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-106                [1, 512, 4, 4]            [1, 512, 4, 4]            2,359,296                  10.03%                   [3, 3]                    37,748,736                True
    │    │    │    │    └─BatchNormAct2d: 5-107        [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                       0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-108                [1, 512, 4, 4]            [1, 2048, 4, 4]           1,048,576                   4.46%                   [1, 1]                    16,777,216                True
    │    │    │    │    └─Identity: 5-109              [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        --
    │    │    │    └─PreActBottleneck: 4-16            [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        True
    │    │    │    │    └─BatchNormAct2d: 5-110        [1, 2048, 4, 4]           [1, 2048, 4, 4]           4,096                       0.02%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-111                [1, 2048, 4, 4]           [1, 512, 4, 4]            1,048,576                   4.46%                   [1, 1]                    16,777,216                True
    │    │    │    │    └─BatchNormAct2d: 5-112        [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                       0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-113                [1, 512, 4, 4]            [1, 512, 4, 4]            2,359,296                  10.03%                   [3, 3]                    37,748,736                True
    │    │    │    │    └─BatchNormAct2d: 5-114        [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                       0.00%                   --                        0                         True
    │    │    │    │    └─Conv2d: 5-115                [1, 512, 4, 4]            [1, 2048, 4, 4]           1,048,576                   4.46%                   [1, 1]                    16,777,216                True
    │    │    │    │    └─Identity: 5-116              [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        --
    ├─BatchNormAct2d: 1-3                              [1, 2048, 4, 4]           [1, 2048, 4, 4]           4,096                       0.02%                   --                        --                        True
    │    └─Identity: 2-7                               [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        --
    │    └─ReLU: 2-8                                   [1, 2048, 4, 4]           [1, 2048, 4, 4]           --                             --                   --                        --                        --
    ├─ClassifierHead: 1-4                              [1, 2048, 4, 4]           [1, 10]                   --                             --                   --                        --                        True
    │    └─SelectAdaptivePool2d: 2-9                   [1, 2048, 4, 4]           [1, 2048, 1, 1]           --                             --                   --                        --                        --
    │    │    └─AdaptiveAvgPool2d: 3-5                 [1, 2048, 4, 4]           [1, 2048, 1, 1]           --                             --                   --                        --                        --
    │    │    └─Identity: 3-6                          [1, 2048, 1, 1]           [1, 2048, 1, 1]           --                             --                   --                        --                        --
    │    └─Dropout: 2-10                               [1, 2048, 1, 1]           [1, 2048, 1, 1]           --                             --                   --                        --                        --
    │    └─Conv2d: 2-11                                [1, 2048, 1, 1]           [1, 10, 1, 1]             20,490                      0.09%                   [1, 1]                    20,490                    True
    │    └─Flatten: 2-12                               [1, 10, 1, 1]             [1, 10]                   --                             --                   --                        --                        --
    =================================================================================================================================================================================================================================
    Total params: 23,520,842
    Trainable params: 23,520,842
    Non-trainable params: 0
    Total mult-adds (Units.MEGABYTES): 995.42
    =================================================================================================================================================================================================================================
    Input size (MB): 0.12
    Forward/backward pass size (MB): 19.47
    Params size (MB): 93.90
    Estimated Total Size (MB): 113.49
    =================================================================================================================================================================================================================================

## Detection models

``` python
from torchvision.models import detection 
import timm
```

``` python
############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter()
###################################################
```

``` python
# Load ResNet model without the final classification layer
model = timm.create_model('resnetv2_50', pretrained=True, num_classes=10)

# weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights,
#                                     box_score_thresh=0.9).train()
```

``` python
import torch

# Generate random input tensor
input_tensor = torch.randn(1, 3, 224, 224)
```

## ONNX

``` python
torch.onnx.export(model, input_tensor, 'Data/resnet50_v2.onnx', input_names=["features"], output_names=["logits"])
```

``` python
############## TENSORBOARD ########################
# writer.add_graph(model, example_data.reshape(-1, 28*28).to(device))

writer.add_graph(model, input_tensor)
writer.flush()
writer.close()
###################################################
```

``` python
list(model.children())[:]
```

    [Sequential(
       (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
       (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
     ),
     Sequential(
       (0): ResNetStage(
         (blocks): Sequential(
           (0): PreActBottleneck(
             (downsample): DownsampleConv(
               (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
               (norm): Identity()
             )
             (norm1): BatchNormAct2d(
               64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (1): PreActBottleneck(
             (norm1): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (2): PreActBottleneck(
             (norm1): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
         )
       )
       (1): ResNetStage(
         (blocks): Sequential(
           (0): PreActBottleneck(
             (downsample): DownsampleConv(
               (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
               (norm): Identity()
             )
             (norm1): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (1): PreActBottleneck(
             (norm1): BatchNormAct2d(
               512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (2): PreActBottleneck(
             (norm1): BatchNormAct2d(
               512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (3): PreActBottleneck(
             (norm1): BatchNormAct2d(
               512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
         )
       )
       (2): ResNetStage(
         (blocks): Sequential(
           (0): PreActBottleneck(
             (downsample): DownsampleConv(
               (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
               (norm): Identity()
             )
             (norm1): BatchNormAct2d(
               512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (1): PreActBottleneck(
             (norm1): BatchNormAct2d(
               1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (2): PreActBottleneck(
             (norm1): BatchNormAct2d(
               1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (3): PreActBottleneck(
             (norm1): BatchNormAct2d(
               1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (4): PreActBottleneck(
             (norm1): BatchNormAct2d(
               1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (5): PreActBottleneck(
             (norm1): BatchNormAct2d(
               1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
         )
       )
       (3): ResNetStage(
         (blocks): Sequential(
           (0): PreActBottleneck(
             (downsample): DownsampleConv(
               (conv): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
               (norm): Identity()
             )
             (norm1): BatchNormAct2d(
               1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (1): PreActBottleneck(
             (norm1): BatchNormAct2d(
               2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
           (2): PreActBottleneck(
             (norm1): BatchNormAct2d(
               2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (norm2): BatchNormAct2d(
               512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
             (norm3): BatchNormAct2d(
               512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
               (drop): Identity()
               (act): ReLU(inplace=True)
             )
             (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
             (drop_path): Identity()
           )
         )
       )
     ),
     BatchNormAct2d(
       2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
       (drop): Identity()
       (act): ReLU(inplace=True)
     ),
     ClassifierHead(
       (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Identity())
       (drop): Dropout(p=0.0, inplace=False)
       (fc): Conv2d(2048, 10, kernel_size=(1, 1), stride=(1, 1))
       (flatten): Flatten(start_dim=1, end_dim=-1)
     )]

Back to top
