from collections import defaultdict

import numpy as np
import torch
from torch import autograd
from options import opt


def freeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    if not net:
        return
    for p in net.parameters():
        p.requires_grad = True

def average_cv_accuracy(cv):
    """
    cv[fold]['valid'] contains CV accuracy for validation set
    cv[fold]['test'] contains CV accuracy for test set
    """
    avg_acc = {'valid': defaultdict(float), 'test': defaultdict(float)}
    for fold, foldacc in cv.items():
        for dataset, cv_acc in foldacc.items():
            for domain, acc in cv_acc.items():
                avg_acc[dataset][domain] += acc
    for domain in avg_acc['valid']:
        avg_acc['valid'][domain] /= opt.kfold
        avg_acc['test'][domain] /= opt.kfold
    # overall average
    return avg_acc

def endless_get_next_batch(loaders, iters, domain):
    try:
        inputs, targets = next(iters[domain])
    except StopIteration:
        iters[domain] = iter(loaders[domain])
        inputs, targets = next(iters[domain])
    # In PyTorch 0.3, Batch Norm no longer works for size 1 batch,
    # so we will skip leftover batch of size < batch_size
    if len(targets) < opt.batch_size:
        return endless_get_next_batch(loaders, iters, domain)
    return (inputs, targets)


domain_labels = {}
def get_domain_label(loss, domain, size):
    if (domain, size) in domain_labels:
        return domain_labels[(domain, size)]
    idx = opt.all_domains.index(domain)
    if loss.lower() == 'l2':
        labels = torch.FloatTensor(size, len(opt.all_domains))
        labels.fill_(-1)
        labels[:, idx].fill_(1)
    else:
        labels = torch.LongTensor(size)
        labels.fill_(idx)
    labels = labels.to(opt.device)
    domain_labels[(domain, size)] = labels
    return labels


random_domain_labels = {}
def get_random_domain_label(loss, size):
    if size in random_domain_labels:
        return random_domain_labels[size]
    labels = torch.FloatTensor(size, len(opt.all_domains))
    if loss.lower() == 'l2':
        labels.fill_(0)
    else:
        labels.fill_(1 / len(opt.all_domains))
    labels = labels.to(opt.device)
    random_domain_labels[size] = labels
    return labels
