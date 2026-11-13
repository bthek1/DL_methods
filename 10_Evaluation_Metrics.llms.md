# Evaluation Metrics

[Evaluation Metrics](https://manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-PerformancecMetrics.html)

Author

Benedict Thekkel

## Accuracy

- Accuracy : is the percentage of correctly predicted examples out of all predictions, formally known as

$Accuracy = \frac{TP + TN}{TP + FP + TN + FN}$

## Precision

- Precision is the the probability of the predicted bounding boxes matching actual ground truth boxes, also referred to as the positive predictive value.

$Precision = \frac{TP}{TP + FP} = \frac{true \ object \ detection}{all \ detected \ boxes}$

## Recall

- Recall is the true positive rate, also referred to as sensitivity, measures the probability of ground truth objects being correctly detected.

$Recall = \frac{TP}{TP + FN} = \frac{true \ object \ detection}{all \ ground \ truth \ boxes}$

## AP - Average Precision

- it is a single number metric that encapsulates both precision and recall and summarizes the Precision-Recall curve by averaging precision across recall values from 0 to 1

$AP = \dfrac{1}{11} \sum_{r \in \{0,0.1,0.2,...,1\}} p_{interp}(r)$

where

$p_{interp}(r) = \max p(\hat{r})$

## mAP (Mean Average Precision)

- count the accumulated TP and the accumulated FP and compute the precision/recall at each line. Average Precision is computed as the average precision at 11 equally spaced recall levels.

Back to top
