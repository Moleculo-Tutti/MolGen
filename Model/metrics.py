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


def pseudo_accuracy_metric_gnn3(model_input, model_output, target, mask, random = True):
    
    num_wanted_cycles = 0
    cycles_created = 0
    good_cycles_created = 0
    cycles_not_created = 0
    cycles_shouldnt_created = 0
    good_types_cycles_predicted = 0

    # Calculate cumulative sums of node counts for each graph
    cumsum_node_counts = model_input.batch.bincount().cumsum(dim=0)
    has_cycle =False
    for i in range(model_input.num_graphs):
        if i == 0:
            # For the first graph, start_index should be 0
            start_index = 0
        else:
            # For the subsequent graphs, start_index is the end_index of the previous graph
            start_index = cumsum_node_counts[i-1]
        
        # end_index is the cumulative sum of node counts up to the current graph
        end_index = cumsum_node_counts[i]

        # check if there is one cycle created in this graph
        current_graph_target = target[start_index:end_index]
        current_graph_output = model_output[start_index:end_index]
        mask_graph = mask[start_index:end_index]

        # compute softmax output 
        current_graph_output_masked = F.softmax(current_graph_output[mask_graph], dim=1)
        sum_on_first_three_dims = current_graph_output_masked[:, :4].sum(dim=1)

        # Trouver l'indice du node avec la plus grande somme
        max_index = torch.argmax(sum_on_first_three_dims)
        # Select in current_graph_output_masked thehighest value 

        vector_predicted = current_graph_output_masked[max_index]
        prediction= torch.multinomial(vector_predicted, 1)

        if torch.sum(current_graph_target[:,:4].max(dim=1)[0]) > 0: 
            # the graph has a cycle
            has_cycle = True
            num_wanted_cycles +=1
        
        
        if has_cycle and prediction < 4 :
            # look if we have predicted one cycle (frist 4  in the vector of 5 ) in this molecul
            cycles_created +=1
            if torch.argmax(current_graph_target[mask_graph][max_index])<4 :
                good_cycles_created += 1
                if prediction == torch.argmax(current_graph_target[mask_graph][max_index]):
                    good_types_cycles_predicted += 1

        if has_cycle and prediction == 4:
            cycles_not_created +=1

        if not(has_cycle) and prediction < 4 :
            cycles_shouldnt_created += 1

        has_cycle = False
        # Set the new feature to 1 for nodes before the 'current_atom
    
    return cycles_created , good_cycles_created , good_types_cycles_predicted , cycles_not_created , cycles_shouldnt_created, num_wanted_cycles

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