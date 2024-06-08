import torch
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


class DBS_Scanner:
    def __init__(self, target_model, benign_model, tokenizer, model_arch, device, config):
        self.target_model = target_model
        self.benign_model = benign_model
        self.tokenizer = tokenizer
        self.device = device
        self.model_arch = model_arch
        self.task = config["task"]

        self.temp = config['init_temp']
        self.max_temp = config['max_temp']
        self.temp_scaling_check_epoch = config['temp_scaling_check_epoch']
        self.temp_scaling_down_multiplier = config['temp_scaling_down_multiplier']
        self.temp_scaling_up_multiplier = config['temp_scaling_up_multiplier']
        self.loss_barrier = config['loss_barrier']
        self.noise_ratio = config['noise_ratio']
        self.rollback_thres = config['rollback_thres']

        self.epochs = config['epochs']
        self.lr = config['lr']
        self.scheduler_step_size = config['scheduler_step_size']
        self.scheduler_gamma = config['scheduler_gamma']

        self.code_max_len = config[self.task]['code_max_len']
        self.nl_max_len = config[self.task]['nl_max_len']
        self.target_len = config['target_len']
        self.trigger_len = config['trigger_len']
        self.eps_to_one_hot = config['eps_to_one_hot']

        self.start_temp_scaling = False
        self.rollback_num = 0
        self.best_asr = 0
        self.best_loss = 1e+10
        self.best_trigger = 'TROJAI_GREAT'
        self.current_trigger = 'TROJAI_GREAT'
        self.best_target = 'TROJAI_GREAT'
        self.current_target = 'TROJAI_GREAT'

        self.placeholder_ids = self.tokenizer.pad_token_id
        self.placeholders = torch.ones(self.trigger_len).to(self.device).long() * self.placeholder_ids
        self.placeholders_attention_mask = torch.ones_like(self.placeholders)
        self.target_placeholders = torch.ones(self.target_len).to(self.device).long() * self.placeholder_ids
        self.target_placeholders_attention_mask = torch.ones_like(self.target_placeholders)
        self.word_embedding = self.target_model.encoder.get_input_embeddings().weight

    def clone_detection_processing(self, sample, max_len, tokenizer):
        code1_tokens = []
        code2_tokens = []

        for i in sample:
            code_tokens1 = " ".join(i["code_tokens1"])
            code1_tokens.append(code_tokens1)

            code_tokens2 = " ".join(i["code_tokens2"])
            code2_tokens.append(code_tokens2)

        code1_tokenizer_outputs = tokenizer(
            code1_tokens, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        code1_ids = code1_tokenizer_outputs["input_ids"]
        code1_masks = code1_tokenizer_outputs["attention_mask"]

        code2_tokenizer_outputs = tokenizer(
            code2_tokens, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        code2_ids = code2_tokenizer_outputs["input_ids"]
        code2_masks = code2_tokenizer_outputs["attention_mask"]

        code_ids = torch.cat((code1_ids, code2_ids), dim=1)
        code_masks = torch.cat((code1_masks, code2_masks), dim=1)

        return code_ids, code_masks

    def defect_detection_processing(self, sample, max_len, tokenizer):
        code_tokens = []

        for i in sample:
            code = " ".join(i["code_tokens"])
            code_tokens.append(code)

        code_tokenizer_outputs = tokenizer(
            code_tokens, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        source_ids = code_tokenizer_outputs["input_ids"]
        source_masks = code_tokenizer_outputs["attention_mask"]

        return source_ids, source_masks

    def code_search_processing(self, samples, code_max_len, nl_max_len):
        codes = []
        nls = []
        for i in samples:
            codes.append(" ".join(i["code_tokens"]))
            nls.append(" ".join(i["docstring_tokens"]))

        code_tokenized_dict = self.tokenizer(
            codes, max_length=code_max_len, padding='max_length', truncation=True, return_tensors='pt')
        code_ids = code_tokenized_dict['input_ids'].to(self.device)
        code_masks = code_tokenized_dict['attention_mask'].to(self.device)

        nl_tokenized_dict = self.tokenizer(
            nls, max_length=nl_max_len, padding='max_length', truncation=True, return_tensors='pt')
        nl_ids = nl_tokenized_dict['input_ids'].to(self.device)
        nl_masks = nl_tokenized_dict['attention_mask'].to(self.device)

        return code_ids, code_masks, nl_ids, nl_masks

    def pre_processing(self, samples, task, code_max_len, nl_max_len, tokenizer, device):
        code_ids, code_masks, nl_ids, nl_masks = None, None, None, None
        if task == "clone_detection":
            code_ids, code_masks = self.clone_detection_processing(samples, code_max_len, tokenizer)
            return code_ids.to(device), code_masks.to(device), None, None
        elif task == "defect_detection":
            code_ids, code_masks = self.defect_detection_processing(samples, code_max_len, tokenizer)
            return code_ids.to(device), code_masks.to(device), None, None
        elif task == "code_search":
            code_ids, code_masks, nl_ids, nl_masks = self.code_search_processing(samples, code_max_len, nl_max_len)
            return code_ids.to(device), code_masks.to(device), nl_ids.to(device), nl_masks.to(device)

    def stamping_placeholder(self, raw_input_ids, raw_attention_masks, insert_idx, max_len, input_type,
                             insert_content=None):
        stamped_input_ids = raw_input_ids.clone()
        stamped_attention_masks = raw_attention_masks.clone()

        insertion_index = torch.zeros(
            raw_input_ids.shape[0]).long().to(self.device)

        placeholders, placeholders_attention_mask = None, None
        if input_type == "code":
            placeholders = self.placeholders
            placeholders_attention_mask = self.placeholders_attention_mask
        elif input_type == "nl":
            placeholders = self.target_placeholders
            placeholders_attention_mask = self.target_placeholders_attention_mask

        if insert_content != None:
            content_attention_mask = torch.ones_like(insert_content)

        for idx in range(raw_input_ids.shape[0]):

            if insert_content == None:

                tmp_input_ids = torch.cat(
                    (
                        raw_input_ids[idx, :insert_idx],
                        placeholders,
                        raw_input_ids[idx, insert_idx:max_len]
                    ), 0
                )[:max_len]

                if tmp_input_ids[-1] != self.tokenizer.pad_token_id:
                    tmp_input_ids[-1] = self.tokenizer.eos_token_id

                tmp_input_ids = torch.cat((tmp_input_ids, raw_input_ids[idx, max_len:]))

                tmp_attention_masks = torch.cat(
                    (
                        raw_attention_masks[idx, :insert_idx],
                        placeholders_attention_mask,
                        raw_attention_masks[idx, insert_idx:max_len]
                    ), 0
                )[:max_len]

                tmp_attention_masks = torch.cat((tmp_attention_masks, raw_attention_masks[idx, max_len:]))

            else:
                pass

            stamped_input_ids[idx] = tmp_input_ids
            stamped_attention_masks[idx] = tmp_attention_masks
            insertion_index[idx] = insert_idx

        return stamped_input_ids, stamped_attention_masks, insertion_index

    def codebert_forward(self, model, stamped_input_ids, sentence_embedding, stamped_attention_masks, max_len, labels):
        if self.task == "clone_detection":
            loss, logits = model(input_embeddings=sentence_embedding, input_mask=stamped_attention_masks,
                                 labels=labels)
            return loss, logits
        elif self.task == "defect_detection":
            loss, logits = model(input_embeddings=sentence_embedding, input_mask=stamped_attention_masks,
                                 labels=labels)
            return loss, logits
        elif self.task == "code_search":
            target_output_embedding = self.target_model.encoder(
                inputs_embeds=sentence_embedding,
                attention_mask=stamped_attention_masks
            )[0]
            target_vec = (target_output_embedding * stamped_attention_masks[:, :, None]).sum(
                1) / stamped_attention_masks.sum(-1)[:, None]
            target_vec = torch.nn.functional.normalize(target_vec, p=2, dim=1)
            return target_vec

    def codet5_forward(self, model, stamped_input_ids, sentence_embedding, stamped_attention_masks, max_len, labels):
        loss, logits = model(input_ids=stamped_input_ids, input_embeddings=sentence_embedding,
                             input_mask=stamped_attention_masks, labels=labels)
        return loss, logits

    def unixcoder_forward(self, model, stamped_input_ids, sentence_embedding, stamped_attention_masks, max_len, labels):
        if self.task == "clone_detection":
            loss, logits = model(input_embeddings=sentence_embedding, input_mask=stamped_attention_masks,
                                 labels=labels)
            return loss, logits
        elif self.task == "defect_detection":
            loss, logits = model(input_embeddings=sentence_embedding, input_mask=stamped_attention_masks,
                                 labels=labels)
            return loss, logits
        elif self.task == "code_search":
            target_output_embedding = self.target_model.encoder(
                inputs_embeds=sentence_embedding,
                attention_mask=stamped_attention_masks
            )[0]
            target_vec = (
                    (target_output_embedding * stamped_attention_masks.ne(0)[:, :, None]).sum(
                        1) / stamped_attention_masks.ne(
                0).sum(-1)[:, None]
            )
            target_vec = torch.nn.functional.normalize(target_vec, p=2, dim=1)
            return target_vec

    def forward(self, epoch, task, inputs, labels, input_type):
        stamped_code_ids, stamped_code_masks, code_insertion_index, \
            stamped_nl_ids, stamped_nl_masks, nl_insertion_index = inputs

        self.optimizer.zero_grad()
        self.target_model.zero_grad()

        opt_var = None
        stamped_input_ids = None
        insertion_index = None
        stamped_input_mask = None

        if input_type == "code":
            opt_var = self.code_opt_var
            stamped_input_ids = stamped_code_ids
            insertion_index = code_insertion_index
            stamped_input_mask = stamped_code_masks
        elif input_type == "nl":
            opt_var = self.nl_opt_var
            stamped_input_ids = stamped_nl_ids
            insertion_index = nl_insertion_index
            stamped_input_mask = stamped_nl_masks

        noise = torch.zeros_like(opt_var).to(self.device)

        # if self.rollback_num >= self.rollback_thres:
        #     self.rollback_num = 0
        #     self.loss_barrier = min(self.loss_barrier * 2, self.best_loss - 1e-3)

        if (epoch) % self.temp_scaling_check_epoch == 0:
            if self.start_temp_scaling:
                if self.ce_loss < self.loss_barrier:
                    self.temp /= self.temp_scaling_down_multiplier

                else:
                    self.rollback_num += 1
                    noise = torch.rand_like(opt_var).to(self.device) * self.noise_ratio
                    self.temp *= self.temp_scaling_down_multiplier
                    if self.temp > self.max_temp:
                        self.temp = self.max_temp
        if input_type == "code":
            self.trigger_bound_opt_var = torch.softmax(opt_var / self.temp + noise, 1)
            trigger_word_embedding = torch.tensordot(self.trigger_bound_opt_var, self.word_embedding, ([1], [0]))
        elif input_type == "nl":
            self.target_bound_opt_var = torch.softmax(opt_var / self.temp + noise, 1)
            trigger_word_embedding = torch.tensordot(self.target_bound_opt_var, self.word_embedding,
                                                     ([1], [0]))

        sentence_embedding = self.target_model.encoder.get_input_embeddings()(stamped_input_ids)

        for idx in range(stamped_input_ids.shape[0]):
            piece1 = sentence_embedding[idx, :insertion_index[idx], :]
            piece2 = sentence_embedding[idx,
                     insertion_index[idx] + opt_var.shape[0]:, :]

            sentence_embedding[idx] = torch.cat(
                (piece1, trigger_word_embedding.squeeze(), piece2), 0)

        if task == 'clone_detection' or task == 'defect_detection':
            loss, logits = eval(f"self.{self.model_arch}_forward")(self.target_model,
                                                                   stamped_input_ids,
                                                                   sentence_embedding,
                                                                   stamped_code_masks,
                                                                   self.code_max_len, labels)

            benign_loss = 0.0
            benign_logits = None
            if self.benign_model != None:
                benign_loss, benign_logits = eval(f"self.{self.model_arch}_forward")(self.benign_model,
                                                                                     sentence_embedding,
                                                                                     stamped_input_mask,
                                                                                     self.code_max_len, labels)

            return loss, logits, benign_loss, benign_logits
        elif task == 'code_search':
            target_vec = eval(f"self.{self.model_arch}_forward")(self.target_model,
                                                                 sentence_embedding,
                                                                 stamped_input_mask,
                                                                 self.code_max_len, labels)
            return target_vec

    def compute_loss(self, logits, benign_logits, labels):
        if self.task == 'code_search':
            code_target_vec = logits
            nl_target_vec = benign_logits
            cur_batch_size = code_target_vec.shape[0]
            loss_func = MSELoss()

            scores = torch.einsum("ab,cb->ac", nl_target_vec, code_target_vec)

            mask = (torch.ones_like(scores) - torch.eye(cur_batch_size, device=scores.device)).bool()
            mask_scores = scores.masked_select(mask).view(cur_batch_size, -1)

            diag = torch.ones([mask_scores.size(0), 1], device=scores.device).to(torch.float32)
            loss = loss_func(mask_scores, diag)

            return loss, mask_scores

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        benign_loss = 0.0
        if benign_logits != None:
            benign_loss = loss_fct(benign_logits, 1 - labels)
        return loss, benign_loss

    def compute_acc(self, logits, labels):
        if self.task == "clone_detection":
            preds = logits[:, 1] > 0.5
        elif self.task == "defect_detection":
            preds = logits[:, 0] > 0.5
        correct = (preds == labels).sum()
        acc = correct / preds.shape[0]
        return acc

    def dim_check(self):

        # extract largest dimension at each position
        values, dims = torch.topk(self.trigger_bound_opt_var, 1, 1)

        # idx = 0
        # dims = topk_dims[:, idx]
        # values = topk_values[:, idx]

        # calculate the difference between current inversion to one-hot
        diff = self.trigger_bound_opt_var.shape[0] - torch.sum(values)

        tmp_trigger = ''
        tmp_trigger_ids = torch.zeros_like(self.placeholders)
        for idy in range(values.shape[0]):
            tmp_trigger = tmp_trigger + ' ' + \
                          self.tokenizer.convert_ids_to_tokens([dims[idy]])[0]
            tmp_trigger_ids[idy] = dims[idy]

        if self.best_loss > self.ce_loss:
            self.best_asr = self.asr
            self.best_loss = self.ce_loss
            self.best_trigger = tmp_trigger
            self.best_trigger_ids = tmp_trigger_ids

        self.current_trigger = tmp_trigger
        # check if current inversion is close to discrete and loss smaller than the bound
        if diff < self.eps_to_one_hot and self.ce_loss <= self.loss_barrier:
            # update best results
            self.best_asr = self.asr
            self.best_loss = self.ce_loss
            self.best_trigger = tmp_trigger
            self.best_trigger_ids = tmp_trigger_ids

            # reduce loss bound to generate trigger with smaller loss
            self.loss_barrier = self.best_loss / 2
            self.rollback_num = 0

        if self.task == "code_search":
            target_values, target_dims = torch.topk(self.target_bound_opt_var, 1, 1)
            target_diff = self.target_bound_opt_var.shape[0] - torch.sum(target_values)
            tmp_target = ''
            tmp_target_ids = torch.zeros_like(self.target_placeholders)
            for idy in range(target_values.shape[0]):
                tmp_target = tmp_target + ' ' + \
                             self.tokenizer.convert_ids_to_tokens([target_dims[idy]])[0]
                tmp_target_ids[idy] = target_dims[idy]
            self.current_target = tmp_target
            # update best results
            self.best_target = tmp_target
            return target_diff, diff

        return diff

    def generate(self, victim_data_list, target_label, position):

        # transform raw text input to tokens
        items = self.pre_processing(victim_data_list,
                                    self.task,
                                    self.code_max_len,
                                    self.nl_max_len,
                                    self.tokenizer,
                                    self.device)

        # get insertion positions
        if position == 'first_half':
            insert_idx = 1

        elif position == 'second_half':
            insert_idx = 50

        # code_ids, code_mask, nl_ids, nl_mask, insert_idx

        # define optimization variable 
        # self.code_opt_var = torch.zeros(self.trigger_len, self.tokenizer.vocab_size).to(self.device)
        self.code_opt_var = torch.zeros(self.trigger_len, self.tokenizer.vocab_size).to(self.device)
        # self.code_opt_var[:, 20815] = 1.0
        self.code_opt_var.requires_grad = True

        # target
        self.nl_opt_var = torch.zeros(self.target_len, self.tokenizer.vocab_size).to(self.device)
        self.nl_opt_var.requires_grad = True
        # TODO
        # self.optimizer = torch.optim.Adam([self.code_opt_var, self.nl_opt_var], lr=self.lr)
        self.optimizer = torch.optim.Adam([self.nl_opt_var, self.code_opt_var], lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            self.scheduler_step_size,
                                                            gamma=self.scheduler_gamma,
                                                            last_epoch=-1)

        # stamping placeholder into the input tokens
        stamped_code_ids, stamped_code_masks, code_insertion_index = None, None, None
        stamped_nl_ids, stamped_nl_masks, nl_insertion_index = None, None, None
        if self.task == "clone_detection" or self.task == "defect_detection" or self.task == "code_search":
            stamped_code_ids, stamped_code_masks, code_insertion_index = self.stamping_placeholder(items[0],
                                                                                                   items[1],
                                                                                                   insert_idx,
                                                                                                   self.code_max_len,
                                                                                                   "code")
        if self.task == "code_search":
            stamped_nl_ids, stamped_nl_masks, nl_insertion_index = self.stamping_placeholder(items[2],
                                                                                             items[3],
                                                                                             insert_idx,
                                                                                             self.nl_max_len,
                                                                                             "nl")

        inputs = (
            stamped_code_ids, stamped_code_masks, code_insertion_index,
            stamped_nl_ids, stamped_nl_masks, nl_insertion_index
        )

        if self.task == "clone_detection" or self.task == "defect_detection":
            for epoch in range(self.epochs):
                target_labels = torch.ones(stamped_code_ids.shape[0]).long().to(
                    self.device) * target_label

                # feed forward
                # logits, benign_logits = self.forward(epoch, self.task, inputs)
                #
                #
                # # compute loss
                # ce_loss, benign_ce_loss = self.compute_loss(logits, benign_logits, target_labels)
                ce_loss, logits, benign_ce_loss, benign_logits = self.forward(epoch, self.task, inputs, target_labels,
                                                                              "code")
                asr = self.compute_acc(logits, target_labels)

                # marginal benign loss penalty
                if epoch == 0:
                    # if benign_asr > 0.75:
                    if isinstance(benign_ce_loss, float):
                        benign_loss_bound = benign_ce_loss
                    else:
                        benign_loss_bound = benign_ce_loss.detach()

                    # else:
                    #     benign_loss_bound = 0.2

                benign_ce_loss = max(benign_ce_loss - benign_loss_bound, 0)

                loss = ce_loss + benign_ce_loss

                if self.model_arch == 'distilbert':
                    loss.backward(retain_graph=True)

                else:
                    loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()

                self.ce_loss = ce_loss
                self.asr = asr

                if ce_loss <= self.loss_barrier:
                    self.start_temp_scaling = True

                self.dim_check()

                print(
                    'Epoch: {}/{}  Loss: {:.4f}  ASR: {:.4f}  Current Trigger: {}  Best Trigger: {}  Best Trigger Loss: {:.4f}  Best Trigger ASR: {:.4f}'.format(
                        epoch, self.epochs, self.ce_loss, self.asr, self.current_trigger, self.best_trigger,
                        self.best_loss,
                        self.best_asr)
                )
        elif self.task == "code_search":
            for epoch in range(self.epochs):
                self.scores = 0
                target_labels = None
                code_target_vec = self.forward(epoch, self.task, inputs, target_labels, "code")
                nl_target_vec = self.forward(epoch, self.task, inputs, target_labels, "nl")
                loss, mask_scores = self.compute_loss(code_target_vec, nl_target_vec, target_labels)
                if self.model_arch == 'distilbert':
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()

                self.scores += np.mean(mask_scores.cpu().detach().numpy())

                self.optimizer.step()
                self.lr_scheduler.step()

                self.ce_loss = loss
                self.asr = mask_scores

                if loss <= self.loss_barrier:
                    self.start_temp_scaling = True
                target_diff, trigger_diff = self.dim_check()
                # trigger_diff = self.dim_check()

                print(
                    'Epoch: {}/{}  Loss: {:.4f}  Mask Scores: {:.4f}  Target Diff: {:.2f}  Trigger Diff: {:.2f}  Current Trigger: {}  Best Target: {}  Best '
                    'Trigger: {}  Best'
                    'Loss: {:.4f}'.format(
                        epoch, self.epochs, self.ce_loss, self.scores, target_diff, trigger_diff,
                        self.current_trigger,
                        self.best_target,
                        self.best_trigger,
                        self.best_loss))

        return self.best_target, self.best_trigger, self.best_loss
