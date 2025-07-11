# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class Model(nn.Module):
    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config

    def forward(self, input_ids=None, input_embeddings=None, input_masks=None, labels=None):
        if input_ids is not None:
            logits = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(0))[0]
        elif input_embeddings is not None:
            logits = self.encoder(inputs_embeds=input_embeddings, attention_mask=input_masks)[0]
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
