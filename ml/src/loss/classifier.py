import torch


def compute_loss(output, label_indices, model, criterion, l2_lambda, device):
    # Calculate cross entropy loss
    ce_loss = criterion(output, label_indices)

    # Calculate L2 regularization loss
    l2_loss = torch.tensor(0.0, requires_grad=True, device=device)
    for param in model.parameters():
        l2_loss = l2_loss + torch.norm(param, p=2)
    l2_loss = l2_lambda * l2_loss

    # Combined loss
    return ce_loss + l2_loss, ce_loss.item(), l2_loss.item()
