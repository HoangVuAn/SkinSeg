import torch
import random
import numpy as np

def fix_all_seed(seed):
    print('seed-----------all device', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def accuracy(y_pred, y_true, threshold=0.5):
    """
    Computes the accuracy between the ground truth labels and the predicted labels.

    Args:
        y_true (torch.Tensor): Ground truth binary or multi-class labels, shape (N,) or (N, C)
        y_pred (torch.Tensor): Predicted labels (probabilities), shape (N,) or (N, C)
        threshold (float): Threshold to binarize y_pred for binary classification, default is 0.5

    Returns:
        float: Accuracy
    """

    y_pred = (y_pred > threshold).float()

    # Calculate the number of correct predictions
    correct_predictions = (y_true == y_pred).sum().item()

    # Calculate the total number of instances
    total_instances = y_true.numel()

    # Calculate accuracy
    accuracy = correct_predictions / total_instances

    return accuracy

def segmentation_metrics(y_pred, y_true, threshold=0.5):
    """
    Computes sensitivity (SEN), specificity (SPE), precision (PRE), and accuracy (ACC) for binary segmentation.
    Args:
        y_true (torch.Tensor): Ground truth binary mask, shape (N, H, W) or (N, 1, H, W)
        y_pred (torch.Tensor): Predicted probabilities or logits, same shape as y_true
        threshold (float): Threshold to binarize predictions
    Returns:
        dict: {'SEN': sensitivity, 'SPE': specificity, 'PRE': precision, 'ACC': accuracy}
    """
    y_pred = (y_pred > threshold).float()
    y_true = y_true.float()
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    SEN = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    SPE = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    PRE = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    ACC = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {'SEN': SEN, 'SPE': SPE, 'PRE': PRE, 'ACC': ACC}

