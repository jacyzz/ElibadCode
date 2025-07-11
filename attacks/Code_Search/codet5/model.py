# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch


class Model(nn.Module):
    # def __init__(self, encoder, config, tokenizer, args):
    def __init__(self, encoder, config):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        # self.tokenizer = tokenizer
        # self.args = args

    def get_t5_vec(self, source_ids):
        # attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        attention_mask = source_ids.ne(0)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        # eos_mask = source_ids.eq(self.tokenizer.eos_token_id)
        eos_mask = source_ids.eq(2)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec
        
    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            # code_inputs = code_inputs.view(-1, self.args.code_length)
            outputs = self.get_t5_vec(code_inputs)
            # outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
            # outputs = (outputs * code_inputs.ne(0)[:, :, None]).sum(1) / code_inputs.ne(0).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            # nl_inputs = nl_inputs.view(-1, self.args.nl_length)
            outputs = self.get_t5_vec(nl_inputs)
            # outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            # outputs = (outputs * nl_inputs.ne(0)[:, :, None]).sum(1) / nl_inputs.ne(0).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
