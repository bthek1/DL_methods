# Transforms

Transforms

Author

Benedict Thekkel

``` python
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('Data/wine.csv', delimiter=",", dtype=np.float32, skiprows = 1)
        self.xy = xy
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
```

``` python
dataset = WineDataset()
```

``` python
first_data = dataset[0]
first_data
```

    (tensor([1.4230e+01, 1.7100e+00, 2.4300e+00, 1.5600e+01, 1.2700e+02, 2.8000e+00,
             3.0600e+00, 2.8000e-01, 2.2900e+00, 5.6400e+00, 1.0400e+00, 3.9200e+00,
             1.0650e+03]),
     tensor([1.]))

``` python
features, labels = dataset[0]
features, labels
```

    (tensor([1.4230e+01, 1.7100e+00, 2.4300e+00, 1.5600e+01, 1.2700e+02, 2.8000e+00,
             3.0600e+00, 2.8000e-01, 2.2900e+00, 5.6400e+00, 1.0400e+00, 3.9200e+00,
             1.0650e+03]),
     tensor([1.]))

``` python
dataloader = DataLoader(dataset=dataset, batch_size = 4, shuffle = True, num_workers = 2)
```

``` python
dataiter = iter(dataloader)
```

``` python
next(dataiter)
```

    [tensor([[1.2250e+01, 1.7300e+00, 2.1200e+00, 1.9000e+01, 8.0000e+01, 1.6500e+00,
              2.0300e+00, 3.7000e-01, 1.6300e+00, 3.4000e+00, 1.0000e+00, 3.1700e+00,
              5.1000e+02],
             [1.2420e+01, 1.6100e+00, 2.1900e+00, 2.2500e+01, 1.0800e+02, 2.0000e+00,
              2.0900e+00, 3.4000e-01, 1.6100e+00, 2.0600e+00, 1.0600e+00, 2.9600e+00,
              3.4500e+02],
             [1.2370e+01, 1.1300e+00, 2.1600e+00, 1.9000e+01, 8.7000e+01, 3.5000e+00,
              3.1000e+00, 1.9000e-01, 1.8700e+00, 4.4500e+00, 1.2200e+00, 2.8700e+00,
              4.2000e+02],
             [1.3830e+01, 1.6500e+00, 2.6000e+00, 1.7200e+01, 9.4000e+01, 2.4500e+00,
              2.9900e+00, 2.2000e-01, 2.2900e+00, 5.6000e+00, 1.2400e+00, 3.3700e+00,
              1.2650e+03]]),
     tensor([[2.],
             [2.],
             [2.],
             [1.]])]

``` python
dataiter = iter(dataloader)
```

``` python
data = next(dataiter)
features, labels = data
```

``` python
features, labels
```

    (tensor([[1.3720e+01, 1.4300e+00, 2.5000e+00, 1.6700e+01, 1.0800e+02, 3.4000e+00,
              3.6700e+00, 1.9000e-01, 2.0400e+00, 6.8000e+00, 8.9000e-01, 2.8700e+00,
              1.2850e+03],
             [1.1840e+01, 2.8900e+00, 2.2300e+00, 1.8000e+01, 1.1200e+02, 1.7200e+00,
              1.3200e+00, 4.3000e-01, 9.5000e-01, 2.6500e+00, 9.6000e-01, 2.5200e+00,
              5.0000e+02],
             [1.3860e+01, 1.5100e+00, 2.6700e+00, 2.5000e+01, 8.6000e+01, 2.9500e+00,
              2.8600e+00, 2.1000e-01, 1.8700e+00, 3.3800e+00, 1.3600e+00, 3.1600e+00,
              4.1000e+02],
             [1.4120e+01, 1.4800e+00, 2.3200e+00, 1.6800e+01, 9.5000e+01, 2.2000e+00,
              2.4300e+00, 2.6000e-01, 1.5700e+00, 5.0000e+00, 1.1700e+00, 2.8200e+00,
              1.2800e+03]]),
     tensor([[1.],
             [2.],
             [2.],
             [1.]]))

``` python
num_epochs = 2
total_samples = len(dataset)
n_iterations = int(np.ceil(total_samples/4))

total_samples, n_iterations
```

    (178, 45)

``` python
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i + 1) % 5 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_iterations}, inputs:{inputs[0][:5]} labels:{labels[0]}')
```

    epoch 1/2, step 5/45, inputs:tensor([13.3600,  2.5600,  2.3500, 20.0000, 89.0000]) labels:tensor([3.])
    epoch 1/2, step 10/45, inputs:tensor([ 13.2900,   1.9700,   2.6800,  16.8000, 102.0000]) labels:tensor([1.])
    epoch 1/2, step 15/45, inputs:tensor([14.1600,  2.5100,  2.4800, 20.0000, 91.0000]) labels:tensor([3.])
    epoch 1/2, step 20/45, inputs:tensor([ 13.9400,   1.7300,   2.2700,  17.4000, 108.0000]) labels:tensor([1.])
    epoch 1/2, step 25/45, inputs:tensor([12.6000,  1.3400,  1.9000, 18.5000, 88.0000]) labels:tensor([2.])
    epoch 1/2, step 30/45, inputs:tensor([ 13.2400,   2.5900,   2.8700,  21.0000, 118.0000]) labels:tensor([1.])
    epoch 1/2, step 35/45, inputs:tensor([11.0300,  1.5100,  2.2000, 21.5000, 85.0000]) labels:tensor([2.])
    epoch 1/2, step 40/45, inputs:tensor([ 13.4800,   1.8100,   2.4100,  20.5000, 100.0000]) labels:tensor([1.])
    epoch 1/2, step 45/45, inputs:tensor([ 12.6400,   1.3600,   2.0200,  16.8000, 100.0000]) labels:tensor([2.])
    epoch 2/2, step 5/45, inputs:tensor([14.7500,  1.7300,  2.3900, 11.4000, 91.0000]) labels:tensor([1.])
    epoch 2/2, step 10/45, inputs:tensor([12.3700,  1.6300,  2.3000, 24.5000, 88.0000]) labels:tensor([2.])
    epoch 2/2, step 15/45, inputs:tensor([ 13.8300,   1.5700,   2.6200,  20.0000, 115.0000]) labels:tensor([1.])
    epoch 2/2, step 20/45, inputs:tensor([12.6900,  1.5300,  2.2600, 20.7000, 80.0000]) labels:tensor([2.])
    epoch 2/2, step 25/45, inputs:tensor([11.4100,  0.7400,  2.5000, 21.0000, 88.0000]) labels:tensor([2.])
    epoch 2/2, step 30/45, inputs:tensor([12.2500,  1.7300,  2.1200, 19.0000, 80.0000]) labels:tensor([2.])
    epoch 2/2, step 35/45, inputs:tensor([ 11.5600,   2.0500,   3.2300,  28.5000, 119.0000]) labels:tensor([2.])
    epoch 2/2, step 40/45, inputs:tensor([ 14.1000,   2.0200,   2.4000,  18.8000, 103.0000]) labels:tensor([1.])
    epoch 2/2, step 45/45, inputs:tensor([ 14.2200,   1.7000,   2.3000,  16.3000, 118.0000]) labels:tensor([1.])

## Dataset Transform

### Types of Transform:

#### On Images:

> CenterCrop, Grayscale, Pad, RandomAffine RandomCrop, RandomHorizontalFlip, RandomRotation Resize, Scale

#### On Tensors:

> LinearTransformation, Normalize, RandomErasing

#### Conversion:

> ToPILImage: from tensor or ndarray

> ToTensor: from numpy.ndarray or PIL Image

``` python
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np

class WineDataset(Dataset):
    def __init__(self, transform = None):
        #data loading
        xy = np.loadtxt('Data/wine.csv', delimiter=",", dtype=np.float32, skiprows = 1)
        self.xy = xy
        self.x = xy[:,1:]
        self.y = xy[:,[0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples
```

``` python
class ToTensor():
    def __call__(self, sample):
        inputs, targets  = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target
```

``` python
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
```

``` python
dataset = WineDataset(transform = composed)
```

``` python
first_data = dataset[0]
first_data
```

    (tensor([2.8460e+01, 3.4200e+00, 4.8600e+00, 3.1200e+01, 2.5400e+02, 5.6000e+00,
             6.1200e+00, 5.6000e-01, 4.5800e+00, 1.1280e+01, 2.0800e+00, 7.8400e+00,
             2.1300e+03]),
     tensor([1.]))

``` python
features, labels = dataset[0]
features, labels
```

    (tensor([5.6920e+01, 6.8400e+00, 9.7200e+00, 6.2400e+01, 5.0800e+02, 1.1200e+01,
             1.2240e+01, 1.1200e+00, 9.1600e+00, 2.2560e+01, 4.1600e+00, 1.5680e+01,
             4.2600e+03]),
     tensor([1.]))

``` python
dataloader = DataLoader(dataset=dataset, batch_size = 4, shuffle = True, num_workers = 2)
```

``` python
dataiter = iter(dataloader)
```

``` python
next(dataiter)
```

    [tensor([[2.7460e+01, 3.0000e+00, 5.4000e+00, 4.5000e+01, 2.0200e+02, 6.0000e+00,
              6.5000e+00, 5.8000e-01, 4.7600e+00, 1.1400e+01, 2.3800e+00, 5.4200e+00,
              2.5700e+03],
             [2.5440e+01, 3.6200e+00, 4.4000e+00, 3.7600e+01, 1.7200e+02, 4.4000e+00,
              5.0600e+00, 5.2000e-01, 3.5400e+00, 7.8000e+00, 2.3200e+00, 6.2800e+00,
              1.4280e+03],
             [2.8200e+01, 4.0400e+00, 4.8000e+00, 3.7600e+01, 2.0600e+02, 5.5000e+00,
              5.8400e+00, 6.4000e-01, 4.7600e+00, 1.2400e+01, 2.1400e+00, 5.5000e+00,
              2.1200e+03],
             [2.6100e+01, 3.5400e+00, 4.2000e+00, 3.4000e+01, 2.1400e+02, 6.0000e+00,
              6.0000e+00, 5.6000e-01, 4.0600e+00, 1.0080e+01, 1.7600e+00, 6.7000e+00,
              1.7700e+03]]),
     tensor([[1.],
             [2.],
             [1.],
             [1.]])]

``` python
dataiter = iter(dataloader)
```

``` python
data = next(dataiter)
features, labels = data
```

``` python
features, labels
```

    (tensor([[2.5200e+01, 2.6800e+00, 3.8000e+00, 3.7000e+01, 1.7600e+02, 2.9000e+00,
              2.7200e+00, 5.8000e-01, 2.7000e+00, 4.9000e+00, 2.0800e+00, 5.5400e+00,
              1.1240e+03],
             [2.6460e+01, 6.6000e+00, 4.5600e+00, 3.7000e+01, 1.9600e+02, 3.6000e+00,
              1.6600e+00, 1.2200e+00, 3.7400e+00, 2.1040e+01, 1.1200e+00, 3.0200e+00,
              1.3500e+03],
             [2.4000e+01, 3.0200e+00, 4.8400e+00, 4.4000e+01, 1.7200e+02, 2.9000e+00,
              2.5000e+00, 1.0000e+00, 3.2600e+00, 7.2000e+00, 2.1000e+00, 5.3000e+00,
              9.0000e+02],
             [2.2820e+01, 1.4800e+00, 5.0000e+00, 4.2000e+01, 1.7600e+02, 4.9600e+00,
              4.0200e+00, 8.4000e-01, 2.8800e+00, 6.1600e+00, 2.2000e+00, 4.6200e+00,
              8.6800e+02]]),
     tensor([[2.],
             [3.],
             [2.],
             [2.]]))

``` python
num_epochs = 2
total_samples = len(dataset)
n_iterations = int(np.ceil(total_samples/4))

total_samples, n_iterations
```

    (178, 45)

``` python
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i + 1) % 5 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_iterations}, inputs:{inputs[0][:5]} labels:{labels[0]}')
```

    epoch 1/2, step 5/45, inputs:tensor([ 24.0000,   6.8600,   4.0000,  38.0000, 174.0000]) labels:tensor([2.])
    epoch 1/2, step 10/45, inputs:tensor([ 26.9800,   3.3200,   4.4800,  48.0000, 174.0000]) labels:tensor([2.])
    epoch 1/2, step 15/45, inputs:tensor([ 25.4000,   7.1000,   4.7200,  43.0000, 212.0000]) labels:tensor([3.])
    epoch 1/2, step 20/45, inputs:tensor([ 22.9200,   7.4800,   3.6400,  39.0000, 214.0000]) labels:tensor([2.])
    epoch 1/2, step 25/45, inputs:tensor([ 23.2800,   4.1200,   4.9200,  43.2000, 168.0000]) labels:tensor([2.])
    epoch 1/2, step 30/45, inputs:tensor([ 27.4400,   2.8600,   5.0000,  33.4000, 216.0000]) labels:tensor([1.])
    epoch 1/2, step 35/45, inputs:tensor([ 23.3000,   3.3400,   5.2400,  52.0000, 176.0000]) labels:tensor([2.])
    epoch 1/2, step 40/45, inputs:tensor([ 26.3400,   5.1800,   4.7400,  40.0000, 240.0000]) labels:tensor([3.])
    epoch 1/2, step 45/45, inputs:tensor([ 29.5000,   3.4600,   4.7800,  22.8000, 182.0000]) labels:tensor([1.])
    epoch 2/2, step 5/45, inputs:tensor([ 27.1600,   5.1600,   5.3800,  49.0000, 210.0000]) labels:tensor([3.])
    epoch 2/2, step 10/45, inputs:tensor([ 26.1000,   3.5400,   4.2000,  34.0000, 214.0000]) labels:tensor([1.])
    epoch 2/2, step 15/45, inputs:tensor([ 24.1400,   4.3200,   4.3400,  42.0000, 170.0000]) labels:tensor([2.])
    epoch 2/2, step 20/45, inputs:tensor([ 24.7400,   3.2600,   4.6000,  49.0000, 176.0000]) labels:tensor([2.])
    epoch 2/2, step 25/45, inputs:tensor([ 24.5800,   2.8200,   3.9600,  32.0000, 170.0000]) labels:tensor([2.])
    epoch 2/2, step 30/45, inputs:tensor([ 24.7400,   2.2600,   4.3200,  38.0000, 174.0000]) labels:tensor([2.])
    epoch 2/2, step 35/45, inputs:tensor([ 26.5600,   3.2800,   5.6800,  31.0000, 220.0000]) labels:tensor([1.])
    epoch 2/2, step 40/45, inputs:tensor([ 24.7400,   1.8800,   2.7200,  21.2000, 176.0000]) labels:tensor([2.])
    epoch 2/2, step 45/45, inputs:tensor([ 26.3400,  10.3800,   4.6400,  44.0000, 186.0000]) labels:tensor([3.])

## Images

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, utils
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from datetime import datetime
from tqdm import tqdm 
import matplotlib.pyplot as plt
import timm
import numpy as np
```

``` python
transform_default = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
])
```

``` python
# Download and load the Imagenette dataset
train_dataset = datasets.Imagenette(root='Data',
                                    split='train',
                                    # download=True,
                                    transform=transform_default,
                                    )


# Download and load the Imagenette dataset
test_dataset = datasets.Imagenette(root='Data',
                                  split='val',
                                  # download=True,
                                  transform=transform_default,
                                 )
```

``` python
# Define a function to display images
def show_image(dataset):
    # Access an image and its label from the dataset
    image, label = dataset
    
    # Convert the image tensor to a NumPy array
    image_np = image[:3].numpy().transpose((1, 2, 0))

    # Display the image using Matplotlib
    plt.imshow(image_np.clip(0,1))
    plt.axis('off')
    plt.title(f' {train_dataset.classes[label][0]}')
    plt.show()
    
   
# Define a function to display images
def show_images(images, labels, **kwargs):
    nrows = int(np.ceil(np.sqrt(len(images))))
    ncols = int(np.ceil(len(images)/nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12),  **kwargs)
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for ax, image, label in zip(axes.flat, images, labels):
        # Convert image to numpy array and adjust pixel values
        img_np = image[:3].numpy().transpose((1, 2, 0))
        
        # Display image
        ax.imshow(img_np.clip(0,1))
        ax.axis('off')
        ax.set_title(f' {train_dataset.classes[label][0]}')
    for ax in axes.flat[len(images):]:
        ax.axis('off')
        
    plt.show()
```

``` python
image, label = train_dataset[2]
```

``` python
type(image)
```

    torch.Tensor

``` python
show_image(train_dataset[2])
```

![](09_Transforms_files/figure-html/cell-34-output-1.png)

## Test Transform

``` python
import torch
import torchvision.transforms.functional as TF

class AddGrayscaleChannel(object):
    def __init__(self):
        super().__init__()
    
    def __call__(self, img):
        # Convert the image to grayscale
        gray_img = TF.rgb_to_grayscale(img)
        
        # Concatenate the grayscale image with the original image along the fourth dimension
        img_with_gray_channel = torch.cat((img, gray_img), dim=0)
        
        return img_with_gray_channel

    def __repr__(self):
        return self.__class__.__name__ + '()'
```

## FFT Transform

``` python
class ComputeFFT(object):
    def __init__(self):
        super().__init__()
    
    def __call__(self, image):
        # Convert the color image to grayscale
        grayscale_image = TF.rgb_to_grayscale(image).squeeze()
        
        # Convert the grayscale image to tensor and apply FFT
        fft_result = torch.fft.fft2(grayscale_image)
        
        # Compute magnitude spectrum
        magnitude_spectrum = torch.log(torch.abs(fft_result) + 1)
        
        # Compute phase spectrum
        phase_spectrum = torch.angle(fft_result)
        
        combined_image = torch.cat((image, magnitude_spectrum.unsqueeze(0), phase_spectrum.unsqueeze(0)), dim=0)

        return combined_image

    def __repr__(self):
        return self.__class__.__name__ + '()'
```

``` python
transform = ComputeFFT()
new_image = transform(image)
transposed_image = torch.transpose(new_image[:3], 0, 2).transpose(0, 1)

plt.imshow(transposed_image, cmap='gray')
plt.title('image')
plt.colorbar()
plt.show()

plt.imshow(new_image[3], cmap='gray')
plt.title('Magnitude Spectrum')
plt.colorbar()
plt.show()

# Visualize phase spectrum
plt.imshow(new_image[4], cmap='gray')
plt.title('Phase Spectrum')
plt.colorbar()
plt.show()
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

![](09_Transforms_files/figure-html/cell-37-output-2.png)

![](09_Transforms_files/figure-html/cell-37-output-3.png)

![](09_Transforms_files/figure-html/cell-37-output-4.png)

``` python
#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
```

    Using cache found in /home/ben/.cache/torch/hub/intel-isl_MiDaS_master

    Loading weights:  None

    Using cache found in /home/ben/.cache/torch/hub/rwightman_gen-efficientnet-pytorch_master
    Using cache found in /home/ben/.cache/torch/hub/intel-isl_MiDaS_master

``` python
class ComputeDepth(object):
    def __init__(self, model, transform, device = "cuda"):
        super().__init__()
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.transform = transform
    
    def __call__(self, image):
        input_image = image.to('cpu').squeeze().numpy().transpose(1, 2, 0)
        # input_image = np.array(image)
        input_batch = transform(input_image).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=input_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
            prediction = prediction.to('cpu')

        prediction_mean = torch.mean(prediction)
        prediction_std = torch.std(prediction)
        
        # Calculate the scaling factors for normalization
        scale_factor = 0.225 / prediction_std
        bias = 0.45 - prediction_mean * scale_factor
        
        # Normalize the tensor to the desired mean and standard deviation
        prediction = prediction * scale_factor + bias
        
        combined_image = torch.cat((image, prediction), dim=0)

        return combined_image

    def __repr__(self):
        return self.__class__.__name__ + '()'
```

``` python
image, label = train_dataset[2]
```

``` python
type(image)
```

    torch.Tensor

``` python
transform_depth = ComputeDepth(midas, transform = transform, device = "cpu")
new_image = transform_depth(image)
transposed_image = torch.transpose(new_image[:3], 0, 2).transpose(0, 1)
```

``` python
plt.imshow(transposed_image, cmap='gray')
plt.title('image')
plt.colorbar()
plt.show()

plt.imshow(new_image[3], cmap='gray')
plt.title('Magnitude Spectrum')
plt.colorbar()
plt.show()
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

![](09_Transforms_files/figure-html/cell-43-output-2.png)

![](09_Transforms_files/figure-html/cell-43-output-3.png)

Back to top
