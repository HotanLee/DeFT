import torch
from torch.nn import functional as F

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10, reduction="mean"):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        if self.reduction == "none":
            loss = self.alpha * ce + self.beta * rce
        else:
            loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class ELRLoss(torch.nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3):
        super(ELRLoss, self).__init__()
        self.num_classes = num_classes
        self.target = torch.zeros(num_examp, self.num_classes).cuda()
        self.beta = beta
        
    def cross_entropy(output, target):
        return F.cross_entropy(output, target)

    def forward(self, index, output, label):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label, reduction="none")
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log())
        final_loss = ce_loss + 5 * elr_reg
        return  final_loss