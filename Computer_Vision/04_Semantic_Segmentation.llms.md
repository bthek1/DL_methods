# Semantic Segmentation

Assigns a class label to each pixel, but does not distinguish between instances of the same class.

Author

Benedict Thekkel

## 1. **Popular Datasets for Semantic Segmentation**

A wide variety of datasets are available for training and evaluating semantic segmentation models. Each dataset has unique characteristics depending on the application domain, such as autonomous driving, general object segmentation, or medical imaging.

### a. **Autonomous Driving Datasets**

1.  **Cityscapes**
    - **Description**: Large-scale dataset for urban scene understanding. Includes street-level imagery from various German cities.
    - **Classes**: 30 classes (e.g., road, person, car, building).
    - **Size**: 5,000 fine-annotated images (2975 train, 500 validation, 1525 test), 20,000 coarse annotations.
    - **Resolution**: 2048×1024 pixels.
    - **Link**: [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
2.  **Mapillary Vistas**
    - **Description**: A diverse dataset collected from various countries, designed for street-level segmentation tasks.
    - **Classes**: 66 object categories.
    - **Size**: 25,000 annotated high-resolution images.
    - **Resolution**: Varies, high resolution.
    - **Link**: [Mapillary Vistas](https://www.mapillary.com/dataset/vistas)
3.  **CamVid**
    - **Description**: One of the earlier datasets for autonomous driving, with video sequences.
    - **Classes**: 32 classes.
    - **Size**: 701 labeled frames.
    - **Resolution**: 960×720 pixels.
    - **Link**: [CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

### b. **General Object Segmentation Datasets**

1.  **PASCAL VOC 2012**
    - **Description**: Widely used dataset for image classification, detection, and segmentation tasks.
    - **Classes**: 21 object categories.
    - **Size**: 1,464 images (train), 1,449 (val), 1,456 (test).
    - **Resolution**: Varies.
    - **Link**: [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
2.  **MS COCO (Common Objects in Context)**
    - **Description**: Large-scale object detection, segmentation, and captioning dataset.
    - **Classes**: 80 object categories.
    - **Size**: 123,287 images with pixel-wise annotations.
    - **Resolution**: Varies.
    - **Link**: [MS COCO](https://cocodataset.org/)

### c. **Medical Imaging Datasets**

1.  **Lung Nodule Analysis (LUNA)**
    - **Description**: A dataset for detecting and segmenting lung nodules in CT scans.
    - **Classes**: Lung nodules.
    - **Size**: 888 CT scans.
    - **Resolution**: Voxel-wise 3D data.
    - **Link**: [LUNA Dataset](https://luna16.grand-challenge.org/)
2.  **BraTS (Brain Tumor Segmentation)**
    - **Description**: A dataset for segmenting brain tumors using MRI scans.
    - **Classes**: Tumor, normal tissue.
    - **Size**: 500+ 3D MRI images.
    - **Resolution**: Voxel-wise 3D data.
    - **Link**: [BraTS Dataset](https://www.med.upenn.edu/sbia/brats2018.html)

------------------------------------------------------------------------

## 2. **Popular Models for Semantic Segmentation**

### a. **Fully Convolutional Networks (FCN)**

- **Architecture**: The first major deep learning-based approach for semantic segmentation. FCNs use convolutional layers for both feature extraction and pixel-wise classification.
- **Key Idea**: Replace fully connected layers with convolutional layers for dense pixel prediction.
- **References**: Long et al., 2015, [FCN Paper](https://arxiv.org/abs/1411.4038)

### b. **U-Net**

- **Architecture**: Symmetrical “U”-shaped network with encoder-decoder architecture.
- **Key Idea**: Encoder extracts features, and the decoder upsamples to full resolution using skip connections to recover spatial information.
- **Applications**: Extremely popular in medical imaging.
- **References**: Ronneberger et al., 2015, [U-Net Paper](https://arxiv.org/abs/1505.04597)

### c. **DeepLab (v1, v2, v3, v3+)**

- **Architecture**: Encoder-decoder architecture with Atrous (dilated) convolutions to capture multi-scale context.
- **Key Idea**: Atrous convolutions and ASPP (Atrous Spatial Pyramid Pooling) enable capturing objects at multiple scales.
- **DeepLab v3+**: Combines spatial pyramid pooling with a decoder module for better object boundary detection.
- **References**: Chen et al., 2017, [DeepLab Paper](https://arxiv.org/abs/1706.05587)

### d. **SegNet**

- **Architecture**: Encoder-decoder architecture that recovers resolution using max-pooling indices from the encoder.
- **Key Idea**: Efficient upsampling using indices from max-pooling in the encoder, which reduces computational cost.
- **References**: Badrinarayanan et al., 2015, [SegNet Paper](https://arxiv.org/abs/1511.00561)

### e. **PSPNet (Pyramid Scene Parsing Network)**

- **Architecture**: Uses a pyramid pooling module to capture multi-scale global context.
- **Key Idea**: Captures global scene-level context using pyramid pooling before the final pixel-wise prediction.
- **References**: Zhao et al., 2017, [PSPNet Paper](https://arxiv.org/abs/1612.01105)

### f. **HRNet (High-Resolution Network)**

- **Architecture**: Maintains high-resolution feature maps throughout the network while using multi-scale fusion.
- **Key Idea**: Avoids the downsampling-heavy nature of many segmentation networks, preserving spatial detail.
- **References**: Wang et al., 2020, [HRNet Paper](https://arxiv.org/abs/1908.07919)

------------------------------------------------------------------------

## 3. **Important Hyperparameters**

Tuning hyperparameters is crucial to improving the performance of segmentation models. Below are the most important ones:

### a. **Learning Rate**

- **Description**: Controls the step size during optimization.
- **Typical Range**: 0.0001 – 0.01 (with learning rate schedules like cosine annealing, or warm restarts).

### b. **Batch Size**

- **Description**: The number of samples processed before the model is updated.
- **Typical Range**: 2 – 16 (for large images due to memory constraints).

### c. **Number of Filters / Feature Maps**

- **Description**: Number of filters in convolutional layers, which controls model capacity.
- **Typical Range**: 32 – 512 per layer, depending on model depth and complexity.

### d. **Optimizer**

- **Popular Choices**:
  - Adam (adaptive learning rate).
  - SGD with momentum (common in large-scale datasets).

### e. **Weight Decay / L2 Regularization**

- **Description**: Helps prevent overfitting by penalizing large weights.
- **Typical Range**: 0.0001 – 0.001.

------------------------------------------------------------------------

## 4. **Popular Loss Functions**

In semantic segmentation, the loss function plays a critical role in training the network by minimizing pixel-wise errors. Common loss functions include:

### a. **Cross-Entropy Loss**

- **Description**: The most common loss for multi-class pixel-wise classification tasks.
- **Formula**: $L = - \sum_{i} y_i \log(\hat{y}_i)$ where \$ y_i \$ is the true label and \$ \_i \$ is the predicted probability.

### b. **Dice Loss**

- **Description**: Measures the overlap between predicted and ground truth segmentation.
- **Formula**: $L = 1 - \frac{2 |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}$ where \$ Y \$ is the ground truth mask, and \$ \$ is the predicted mask.

### c. **Intersection over Union (IoU) Loss**

- **Description**: Measures the overlap between the predicted and ground truth areas.
- **Formula**: $\text{IoU} = \frac{|Y \cap \hat{Y}|}{|Y \cup \hat{Y}|}$ where \$ \|Y\| \$ is the area of ground truth and \$ \|\| \$ is the area of prediction.

### d. **Tversky Loss**

- **Description**: A generalization of Dice Loss, controlling false positives and false negatives.
- **Formula**: \$ L = 1 - \$

------------------------------------------------------------------------

## 5. **Other Important Topics**

### a. **Data Augmentation**

- **Techniques**: Random crop, horizontal/vertical flipping, color jittering, and elastic deformation.
- **Purpose**: Prevent overfitting and improve generalization.

### b. **Post-Processing Techniques**

- **CRF (Conditional Random Field)**: Often used as a post-processing step to refine segmentation boundaries by enforcing spatial consistency.

### c. **Evaluation Metrics**

- **Mean Intersection over Union (mIoU)**: The most widely used evaluation metric for segmentation tasks.
- **Pixel Accuracy**: The ratio of correctly predicted pixels to total pixels.

------------------------------------------------------------------------

## 6. **References**

1.  Long, J., Shelhamer, E., & Darrell, T. (2015). “Fully Convolutional Networks for Semantic Segmentation.” CVPR. [Link](https://arxiv.org/abs/1411.4038)
2.  Ronneberger, O., Fischer, P., & Brox, T. (2015). “U-Net: Convolutional Networks for Biomedical Image Segmentation.” MICCAI. [Link](https://arxiv.org/abs/1505.04597)
3.  Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017). “DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs.” PAMI. [Link](https://arxiv.org/abs/1606.00915)
4.  Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). “Pyramid Scene Parsing Network.” CVPR. [Link](https://arxiv.org/abs/1612.01105)
5.  Wang, J., Sun, K., Cheng, T., Jiang, B., Deng, C., Zhao, Y., & Liu, W. (2020). “Deep High-Resolution Representation Learning for Visual Recognition.” PAMI. [Link](https://arxiv.org/abs/1908.07919)

------------------------------------------------------------------------

Back to top
