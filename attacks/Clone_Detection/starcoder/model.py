# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class Model(nn.Module):
    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, input_ids=None, labels=None):
        logits = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(0))[0]
        prob = F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
