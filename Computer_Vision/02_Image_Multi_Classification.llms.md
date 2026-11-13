# Image Multi Classification

The task of assigning multiple labels to an entire image, indicating what object or scene is present in the image.

Author

Benedict Thekkel

## Popular Datasets

1.  CIFAR-10

- **Description**: 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- **URL**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

2.  CIFAR-100

- **Description**: 60,000 32x32 color images in 100 classes, with 600 images per class.
- **URL**: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

3.  ImageNet

- **Description**: Over 14 million images categorized into over 20,000 classes.
- **URL**: [ImageNet](http://www.image-net.org/)

4.  MNIST

- **Description**: 70,000 28x28 grayscale images of handwritten digits in 10 classes.
- **URL**: [MNIST](http://yann.lecun.com/exdb/mnist/)

5.  Fashion-MNIST

- **Description**: 70,000 28x28 grayscale images of 10 fashion categories.
- **URL**: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

## Popular Models

1.  Convolutional Neural Networks (CNNs)

- **Examples**: LeNet, AlexNet, VGG, ResNet, DenseNet

2.  Residual Networks (ResNet)

- **Description**: Introduces skip connections to prevent vanishing gradients.
- **URL**: [ResNet](https://arxiv.org/abs/1512.03385)

3.  DenseNet

- **Description**: Connects each layer to every other layer in a feed-forward fashion.
- **URL**: [DenseNet](https://arxiv.org/abs/1608.06993)

4.  Inception Networks (GoogLeNet)

- **Description**: Uses multiple types of convolutions in parallel to capture different features.
- **URL**: [Inception](https://arxiv.org/abs/1409.4842)

5.  EfficientNet

- **Description**: Scales up model size while balancing network depth, width, and resolution.
- **URL**: [EfficientNet](https://arxiv.org/abs/1905.11946)

## Hyperparameters

1.  Learning Rate

- **Description**: Controls how much to change the model in response to the estimated error each time the model weights are updated.

2.  Batch Size

- **Description**: The number of training examples utilized in one iteration.

3.  Number of Epochs

- **Description**: The number of complete passes through the training dataset.

4.  Optimizer

- **Examples**: SGD, Adam, RMSprop

5.  Weight Initialization

- **Examples**: Xavier, He initialization

6.  Dropout Rate

- **Description**: Fraction of the input units to drop.

7.  Learning Rate Decay

- **Description**: Reduces the learning rate as training progresses.

## Popular Loss Functions

1.  Cross-Entropy Loss

- **Description**: Measures the performance of a classification model whose output is a probability value between 0 and 1.

2.  Hinge Loss

- **Description**: Used for “maximum-margin” classification, mostly for SVMs.

3.  Kullback-Leibler Divergence Loss

- **Description**: Measures how one probability distribution diverges from a second, expected probability distribution.

## Popular Evaluation Metrics

1.  Accuracy

- **Description**: The ratio of correctly predicted observation to the total observations.

2.  Precision

- **Description**: The ratio of correctly predicted positive observations to the total predicted positives.

3.  Recall (Sensitivity)

- **Description**: The ratio of correctly predicted positive observations to the all observations in actual class.

4.  F1 Score

- **Description**: The weighted average of Precision and Recall.

5.  Confusion Matrix

- **Description**: A table used to describe the performance of a classification model.

6.  ROC-AUC

- **Description**: A graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

## Other Important Topics

1.  Data Augmentation

- **Description**: Techniques used to increase the diversity of your training data without actually collecting new data.
- **Examples**: Rotation, Translation, Scaling, Flipping, Adding Noise

2.  Transfer Learning

- **Description**: Reusing a pre-trained model on a new problem.
- **Example**: Fine-tuning a model pre-trained on ImageNet for a new classification task.

3.  Regularization Techniques

- **Examples**: L1/L2 regularization, Dropout, Early Stopping

4.  Ensemble Methods

- **Description**: Combining the predictions of multiple models to improve generalizability and robustness.
- **Examples**: Bagging, Boosting, Stacking

5.  Fine-Tuning

- **Description**: Adjusting the parameters of an existing model to better fit the new data.

6.  Hyperparameter Tuning

- **Techniques**: Grid Search, Random Search, Bayesian Optimization

7.  Model Interpretability

- **Techniques**: SHAP values, LIME

Back to top
