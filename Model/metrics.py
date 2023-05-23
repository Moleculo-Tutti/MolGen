import torch
import torch.nn.functional as F

import torch.nn as nn

def pseudo_accuracy_metric(model_output, target, random = False):

    if random:
        # draw a random label according to the distribution of the model output
        model_output = torch.multinomial(model_output, 1)
    else:
        model_output = model_output.argmax(dim=1)
    # for each element in the batch, if the model output predited is none 0 for the target class, then it is correct
    
    correct = 0
    for i in range(len(model_output)):
        if target[i][model_output[i]] > 0.01:
            correct += 1

    return correct

def pseudo_recall_for_each_class(model_output, target, random = False):
    encoding_size = len(model_output[0])
    if random:
        # draw a random label according to the distribution of the model output
        model_output = torch.multinomial(model_output, 1)
    else:
        model_output = model_output.argmax(dim=1)
    # for each element in the batch, if the model output predited is none 0 for the target class, then it is correct

    correct = torch.zeros(encoding_size)
    count_per_class = torch.zeros(encoding_size)
    for i in range(len(model_output)):
        if target[i][model_output[i]] > 0.01:
            correct[model_output[i]] += 1
            count_per_class[model_output[i]] += 1
        else:
            count_per_class[target[i].argmax(dim=0)] += 1

    return correct, count_per_class


def pseudo_precision_for_each_class(model_output, target, random=False):
    encoding_size = len(model_output[0])
    if random:
        # draw a random label according to the distribution of the model output
        model_output = torch.multinomial(model_output, 1)
    else:
        model_output = model_output.argmax(dim=1)
    # for each element in the batch, if the model output predited is none 0 for the target class, then it is correct
    
    correct = torch.zeros(encoding_size)
    count_per_class = torch.zeros(encoding_size)
    for i in range(len(model_output)):
        if target[i][model_output[i]] > 0.01:
            correct[model_output[i]] += 1
        count_per_class[model_output[i]] += 1

    return correct, count_per_class


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target, mask):
        loss = self.loss_fn(output, target)
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()


class FocalLoss(torch.nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is None:
            self.alpha = torch.ones(self.num_classes, 1) / num_classes
        else:
            self.alpha = torch.tensor(alpha).view(self.num_classes, 1)

    def forward(self, inputs, targets):
        log_softmax = F.log_softmax(inputs, dim=1)
        targets = targets.to(torch.float32)  # Convert targets to float32
        logpt = (log_softmax * targets).sum(dim=1, keepdim=True)
        pt = torch.exp(logpt)
        
        alpha_t = self.alpha.to(inputs.device).view(1, -1)
        alpha_t = (alpha_t * targets).sum(dim=1, keepdim=True)

        loss = -alpha_t * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss