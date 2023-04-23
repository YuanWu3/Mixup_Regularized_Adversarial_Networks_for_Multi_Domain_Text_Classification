import torch
import torch.nn.functional as functional
from torch import autograd, nn

from options import opt


class MlpFeatureExtractor(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(MlpFeatureExtractor, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.net = nn.Sequential()
        num_layers = len(hidden_sizes)
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('f-linear-{}'.format(i), nn.Linear(input_size, hidden_sizes[0]))
            else:
                self.net.add_module('f-linear-{}'.format(i), nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            if batch_norm:
                self.net.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_sizes[i]))
            self.net.add_module('f-relu-{}'.format(i), opt.act_unit)

        if dropout > 0:
            self.net.add_module('f-dropout-final', nn.Dropout(p=dropout))
        self.net.add_module('f-linear-final', nn.Linear(hidden_sizes[-1], output_size))
        if batch_norm:
            self.net.add_module('f-bn-final', nn.BatchNorm1d(output_size))
        self.net.add_module('f-relu-final', opt.act_unit)

    def forward(self, input):
        return self.net(input)

class SentimentClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(SentimentClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.hidden_size = hidden_size
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-softmax', nn.Softmax(dim=-1))

    def forward(self, input):
        return self.net(input)


class DomainClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 num_domains,
                 loss_type,
                 dropout,
                 batch_norm=False):
        super(DomainClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.num_domains = num_domains
        self.loss_type = loss_type
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.net.add_module('q-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
            else:
                self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), opt.act_unit)

        self.net.add_module('q-linear-final', nn.Linear(hidden_size, num_domains))
        if loss_type.lower() == 'gr' or loss_type.lower() == 'bs':
            self.net.add_module('q-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        scores = self.net(input)
        if self.loss_type.lower() == 'l2':
            # normalize
            scores = functional.relu(scores)
            scores /= torch.sum(scores, dim=1, keepdim=True)
        return scores
