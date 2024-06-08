# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model * 2, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out_proj = nn.Linear(config.d_model, 2)

    def forward(self, features, **kwargs):
        x = features  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.classifier = RobertaClassificationHead(config)

    def get_t5_vec(self, input_ids, input_embeddings = None, input_mask = None, block_size=400):
        # attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        input_ids = input_ids.view(-1, block_size)
        if input_ids != None and input_embeddings == None and input_mask == None:
            input_ids = input_ids.view(-1, block_size)
            attention_mask = input_ids.ne(0)
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        elif input_embeddings != None and input_mask != None:
            input_embeddings = input_embeddings.view(-1, block_size, input_embeddings.shape[-1])
            input_mask = input_mask.view(-1, block_size)
            outputs = self.encoder(
                inputs_embeds=input_embeddings,
                attention_mask=input_mask,
                labels=input_ids,
                decoder_attention_mask=input_mask,
                output_hidden_states=True
            )
        hidden_states = outputs['decoder_hidden_states'][-1]
        # eos_mask = source_ids.eq(self.tokenizer.eos_token_id)
        eos_mask = input_ids.eq(2)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, input_ids = None, input_embeddings = None, input_mask = None, labels = None, block_size=400):
        outputs = self.get_t5_vec(input_ids, input_embeddings, input_mask, block_size)
        logits = self.classifier(outputs)
        prob = F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
