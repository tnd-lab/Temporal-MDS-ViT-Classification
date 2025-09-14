import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        """
        Focal Loss for Multi-class Classification

        Args:
            alpha (float): Weighting factor for class imbalance.
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Raw model outputs (logits), shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].

        Returns:
            Tensor: Computed focal loss.
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)  # [batch_size, num_classes]

        # Gather probabilities of the true class
        targets_one_hot = F.one_hot(
            targets, num_classes=logits.size(1)
        )  # [batch_size, num_classes]
        probs = torch.sum(probs * targets_one_hot, dim=1)  # [batch_size]

        # Compute focal loss
        log_probs = -torch.log(probs + 1e-9)  # Stability for log
        focal_term = (1 - probs) ** self.gamma
        loss = self.alpha * focal_term * log_probs

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
