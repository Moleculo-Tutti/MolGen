import torch

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
    if random:
        # draw a random label according to the distribution of the model output
        model_output = torch.multinomial(model_output, 1)
    else:
        model_output = model_output.argmax(dim=1)
    # for each element in the batch, if the model output predited is none 0 for the target class, then it is correct
    
    correct = torch.zeros(10)
    count_per_class = torch.zeros(10)
    for i in range(len(model_output)):
        if target[i][model_output[i]] > 0.01:
            correct[model_output[i]] += 1
            count_per_class[model_output[i]] += 1
        else:
            count_per_class[target[i].argmax(dim=0)] += 1

    return correct, count_per_class


def pseudo_precision_for_each_class(model_output, target, random=False):

    if random:
        # draw a random label according to the distribution of the model output
        model_output = torch.multinomial(model_output, 1)
    else:
        model_output = model_output.argmax(dim=1)
    # for each element in the batch, if the model output predited is none 0 for the target class, then it is correct
    
    correct = torch.zeros(10)
    count_per_class = torch.zeros(10)
    for i in range(len(model_output)):
        if target[i][model_output[i]] > 0.01:
            correct[model_output[i]] += 1
        count_per_class[model_output[i]] += 1

    return correct, count_per_class


