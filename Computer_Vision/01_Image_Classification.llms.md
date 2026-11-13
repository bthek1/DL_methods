# Image classification

The task of assigning a label to an entire image, indicating what object or scene is present in the image.

Author

Benedict Thekkel

## Dataset

- CIFAR-10 and CIFAR-100
  - CIFAR-10: 60,000 32x32 color images in 10 classes, with 6,000 images per class.
  - CIFAR-100: Similar to CIFAR-10 but with 100 classes containing 600 images each.
  - Use Case: Benchmarking small-scale image classification models.
- ImageNet
  - Description: Over 14 million images across 1,000 classes.
  - Use Case: Large-scale image classification, used for pre-training models.
- MNIST and Fashion-MNIST
  - MNIST: 70,000 grayscale images of handwritten digits (0-9).
  - Fashion-MNIST: 70,000 grayscale images of fashion items in 10 categories.
  - Use Case: Benchmarking simple image classification models.
- COCO (Common Objects in Context)
  - Description: 330,000 images with objects segmented into 80 categories.
  - Use Case: Object detection, segmentation, and image classification.
- SVHN (Street View House Numbers)
  - Description: Over 600,000 32x32 color images of house numbers from Google Street View.
  - Use Case: Real-world digit classification.

## Models

### Convolutional Neural Networks (CNNs)

- LeNet: One of the earliest CNN architectures, used for digit recognition.
- AlexNet: Won the 2012 ImageNet competition, significantly deep with ReLU activations.
- VGGNet: Known for using very small (3x3) convolution filters, 16-19 weight layers.
- GoogLeNet (Inception): Uses a network-in-network architecture, significantly reducing parameters.
- ResNet (Residual Networks): Introduces residual connections to train very deep networks.
- DenseNet: Each layer receives input from all previous layers, promoting feature reuse.

### Vision Transformers (ViT)

- Description: Adapted the Transformer architecture to image classification tasks.
- Use Case: Competes with CNNs on image classification benchmarks.

### EfficientNet

- Description: Uses a compound scaling method to uniformly scale width, depth, and resolution.
- Use Case: Balances accuracy and efficiency, outperforming many existing models.

## Hyperparameters

1.  Learning Rate

- Description: Controls the step size at each iteration while moving towards a minimum of the loss function.
- Tuning: Start with a moderate value (e.g., 0.001), use learning rate schedules (e.g., step decay, exponential decay).

2.  Batch Size

- Description: Number of samples processed before the model is updated.
- Tuning: Common values range from 32 to 256. Larger batches require more memory but can leverage better parallelism.

4.  Number of Epochs

- Description: Number of complete passes through the training dataset.
- Tuning: Monitor validation loss to avoid overfitting, typically between 10 to 100 epochs.

5.  Optimizer

- Popular Choices: SGD, Adam, RMSprop.
- Tuning: Adam is a good default choice; try SGD with momentum for potentially better convergence.

6.  Regularization Parameters

- Weight Decay (L2 Regularization): Penalizes large weights, reducing overfitting.
- Dropout Rate: Randomly sets a fraction of input units to 0 at each update during training, preventing overfitting.

## Loss Functions

1.  Cross-Entropy Loss

- Description: Measures the performance of a classification model whose output is a probability value between 0 and 1.
- Use Case: Standard for multi-class classification problems.

2.  Mean Squared Error (MSE)

- Description: Measures the average of the squares of the errors between predicted and actual values.
- Use Case: More common in regression, but sometimes used in classification problems with continuous labels.

4.  Categorical Hinge Loss

- Description: Measures the performance for “one-versus-all” classification tasks.
- Use Case: Useful in scenarios with class imbalance.

5.  Focal Loss

- Description: Modifies cross-entropy loss to address class imbalance by down-weighting the loss assigned to well-classified examples.
- Use Case: Effective in highly imbalanced datasets.

### Other Important Topics

- Data Augmentation
  - Description: Techniques to artificially increase the size of a dataset by creating modified versions of images.
  - Methods: Rotation, flipping, scaling, cropping, color jittering.
  - Use Case: Helps improve model generalization.
- Transfer Learning
  - Description: Using a pre-trained model on a new, but related task.
  - Approach: Fine-tuning the pre-trained model on the new dataset.
  - Use Case: Effective when the new dataset is small.
- Evaluation Metrics
  - Accuracy: Proportion of correctly predicted instances.
  - Precision, Recall, F1-Score: Useful for class imbalance scenarios.
  - Confusion Matrix: Provides insight into the performance of the classification model on each class.
- Model Interpretability
  - Grad-CAM (Gradient-weighted Class Activation Mapping): Visualizes which parts of the image are important for the model’s predictions.
  - SHAP (SHapley Additive exPlanations): Explains the output of machine learning models.

Back to top
