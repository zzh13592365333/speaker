import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            loss = alpha.gather(0, targets.view(-1)) * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, feat_a, feat_b):
        feat_a = F.normalize(feat_a, dim=1)
        feat_b = F.normalize(feat_b, dim=1)
        logits = torch.matmul(feat_a, feat_b.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=feat_a.device)
        return F.cross_entropy(logits, labels, reduction=self.reduction)


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()
