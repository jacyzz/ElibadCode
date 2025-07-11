import sys

import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np

from tqdm import tqdm
import random


class Scanner:
    def __init__(self, task, target_model, benign_model, tokenizer, selected_vocab_idxes, model_arch, device, config):
        self.target_model = target_model
        self.benign_model = benign_model
        self.tokenizer = tokenizer
        self.device = device
        self.task = task
        self.model_arch = model_arch

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

        self.code_max_len = config[task]['code_max_len']
        self.nl_max_len = config[task]['nl_max_len']
        self.trigger_len = config['trigger_len']
        self.target_len = config['target_len']
        self.eps_to_one_hot = config['eps_to_one_hot']

        self.topk = config['topk']
        self.repeat_size = config['repeat_size']
        self.batch_size = config['batch_size']

        self.start_temp_scaling = False
        self.rollback_num = 0
        self.best_asr = 0
        self.best_loss = 1e+10
        self.best_trigger = 'TROJAI_GREAT'
        self.current_trigger = 'TROJAI_GREAT'
        self.visited_trigger_ids = torch.tensor([]).to(self.device)
        self.visited_target_ids = torch.tensor([]).to(self.device)

        self.placeholder_ids = self.tokenizer.pad_token_id
        self.placeholders = torch.ones(self.trigger_len).to(self.device).long() * self.placeholder_ids
        self.placeholders_attention_mask = torch.ones_like(self.placeholders)
        self.nl_placeholders = torch.ones(self.target_len).to(self.device).long() * self.placeholder_ids
        self.nl_placeholders_attention_mask = torch.ones_like(self.nl_placeholders)

        self.word_embedding = self.target_model.encoder.get_input_embeddings().weight

        self.selected_vocab_idxes = selected_vocab_idxes

        if len(selected_vocab_idxes):
            selected_vocab_idxes = torch.tensor(selected_vocab_idxes).to(device)
            self.selected_word_embedding = torch.index_select(self.word_embedding, 0, selected_vocab_idxes)
        else:
            self.selected_word_embedding = self.word_embedding.clone().detach()

    def bigclonebench_processing(self, sample, max_len):
        code1_tokens = []
        code2_tokens = []

        for i in sample:
            code_tokens1 = " ".join(i["code_tokens1"])
            code1_tokens.append(code_tokens1)

            code_tokens2 = " ".join(i["code_tokens2"])
            code2_tokens.append(code_tokens2)


        code1_tokenizer_outputs = self.tokenizer(
            code1_tokens, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        code1_ids = code1_tokenizer_outputs["input_ids"]
        code1_mask = code1_tokenizer_outputs["attention_mask"]

        code2_tokenizer_outputs = self.tokenizer(
            code2_tokens, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        code2_ids = code2_tokenizer_outputs["input_ids"]
        code2_mask = code2_tokenizer_outputs["attention_mask"]

        source_ids = torch.cat((code1_ids, code2_ids), dim=1).to(self.device)
        source_mask = torch.cat((code1_mask, code2_mask), dim=1).to(self.device)

        return source_ids, source_mask

    def devign_processing(self, sample, max_len):
        code_tokens = []

        for i in sample:
            code = " ".join(i["code_tokens"])
            code_tokens.append(code)

        code_tokenizer_outputs = self.tokenizer(
            code_tokens, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        source_ids = code_tokenizer_outputs["input_ids"].to(self.device)
        source_mask = code_tokenizer_outputs["attention_mask"].to(self.device)

        return source_ids, source_mask

    def codesearchnet_processing(self, sample, max_len, key):
        source = []

        for i in sample:
            source.append(" ".join(i[key]))

        source_tokenized_outputs = self.tokenizer(
            source, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        source_ids = source_tokenized_outputs['input_ids'].to(self.device)
        source_mask = source_tokenized_outputs['attention_mask'].to(self.device)

        return source_ids, source_mask

    def pre_processing(self, task, sample, max_len, key):
        if task == "clone_detection":
            source_ids, source_mask = self.bigclonebench_processing(sample, max_len)
        elif task == "defect_detection":
            source_ids, source_mask = self.devign_processing(sample, max_len)
        elif task == "code_search":
            source_ids, source_mask = self.codesearchnet_processing(sample, max_len, key)

        return source_ids, source_mask

    def generate_trigger(self, trigger_len):
        gt_ids = np.arange(len(self.selected_vocab_idxes))
        trigger = []
        trigger_ids = []
        for i in range(trigger_len):
            gt_id = np.random.choice(gt_ids)
            trigger.append(self.tokenizer.decode([self.selected_vocab_idxes[gt_id]]))
            trigger_ids.append(gt_id)
        trigger = "_".join(trigger)

        return trigger, trigger_ids

    def stamping_placeholder(self, raw_input_ids, raw_attention_mask, insert_idx, max_len,
                             placeholders, placeholders_attention_mask, insert_content=None):
        stamped_input_ids = raw_input_ids.clone()
        stamped_attention_mask = raw_attention_mask.clone()

        insertion_index = torch.zeros(
            raw_input_ids.shape[0]).long().to(self.device)

        if insert_content != None:
            content_attention_mask = torch.ones_like(insert_content)

        for idx in range(raw_input_ids.shape[0]):

            if insert_content == None:

                tmp_input_ids = torch.cat(
                    (
                        raw_input_ids[idx, :insert_idx[idx]],
                        # raw_input_ids[idx, :insert_idx],
                        placeholders,
                        raw_input_ids[idx, insert_idx[idx]:max_len]
                        # raw_input_ids[idx, insert_idx:max_len]
                    ), 0
                )[:max_len]

                if tmp_input_ids[-1] != self.tokenizer.pad_token_id:
                    tmp_input_ids[-1] = self.tokenizer.eos_token_id

                tmp_input_ids = torch.cat((tmp_input_ids, raw_input_ids[idx, max_len:]))

                tmp_attention_mask = torch.cat(
                    (
                        raw_attention_mask[idx, :insert_idx[idx]],
                        # raw_attention_mask[idx, :insert_idx],
                        placeholders_attention_mask,
                        raw_attention_mask[idx, insert_idx[idx]:max_len]
                        # raw_attention_mask[idx, insert_idx:max_len]
                    ), 0
                )[:max_len]

                tmp_attention_mask = torch.cat((tmp_attention_mask, raw_attention_mask[idx, max_len:]))

            else:
                pass

            stamped_input_ids[idx] = tmp_input_ids
            stamped_attention_mask[idx] = tmp_attention_mask
            insertion_index[idx] = insert_idx[idx]

        return stamped_input_ids, stamped_attention_mask, insertion_index

    def codebert_forward(self, model, stamped_input_ids, sentence_embedding, stamped_attention_mask, max_len, labels):
        if self.task == "clone_detection":
            loss, logits = model(input_embeddings=sentence_embedding, input_mask=stamped_attention_mask, labels=labels)
            return loss, logits
        elif self.task == "defect_detection":
            loss, logits = model(input_embeddings=sentence_embedding, input_mask=stamped_attention_mask, labels=labels)
            return loss, logits
        elif self.task == "code_search":
            source_vec = model(source_embeddings=sentence_embedding, source_mask=stamped_attention_mask)
            return source_vec

    def codet5_forward(self, model, stamped_input_ids, sentence_embedding, stamped_attention_masks, max_len, labels):
        loss, logits = model(input_ids=stamped_input_ids, input_embeddings=sentence_embedding,
                             input_mask=stamped_attention_masks, labels=labels)
        return loss, logits

    def unixcoder_forward(self, model, stamped_input_ids, sentence_embedding, stamped_attention_mask, max_len, labels):
        if self.task == "clone_detection":
            loss, logits = model(input_embeddings=sentence_embedding, input_mask=stamped_attention_mask, labels=labels)
            return loss, logits
        elif self.task == "defect_detection":
            loss, logits = model(input_embeddings=sentence_embedding, input_mask=stamped_attention_mask, labels=labels)
            return loss, logits
        elif self.task == "code_search":
            source_vec = model(source_embeddings=sentence_embedding, source_mask=stamped_attention_mask)
            return source_vec

    def forward(self, stamped_input_ids, stamped_attention_mask,
                trigger_embeddings, insertion_index, max_len, trigger_len, labels):
        sentence_embedding = self.target_model.encoder.get_input_embeddings()(stamped_input_ids)

        for idx in range(stamped_input_ids.shape[0]):
            piece1 = sentence_embedding[idx, :insertion_index[idx], :]
            piece2 = sentence_embedding[idx,
                     insertion_index[idx] + trigger_len:, :]

            sentence_embedding[idx] = torch.cat(
                (piece1, trigger_embeddings, piece2), 0)

        if self.task == "code_search":
            source_vec = eval(f"self.{self.model_arch}_forward")(self.target_model,
                                                                 sentence_embedding,
                                                                 stamped_attention_mask,
                                                                 max_len, labels)
            return source_vec
        else:
            loss, logits = eval(f"self.{self.model_arch}_forward")(self.target_model,
                                                                   stamped_input_ids,
                                                                   sentence_embedding,
                                                                   stamped_attention_mask,
                                                                   max_len,
                                                                   labels)
            benign_loss = 0.0
            benign_logits = None

            return loss, logits, benign_loss, benign_logits

    def compute_mse_loss(self, code_vec, nl_vec):
        loss_fct = MSELoss()
        scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)

        diag = torch.ones([scores.size(0), 1], device=scores.device).to(torch.float32)
        loss = loss_fct(scores, diag)

        return loss, scores

    def compute_loss(self, logits, benign_logits, labels):

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

    def generate_candidate(self, grad, trigger_ids, visited_candidate_ids, trigger_len, source_ids, insert_idx,
                           max_len):
        top_indices = (-grad).topk(self.topk, dim=1).indices
        top_indices = torch.tensor([[self.selected_vocab_idxes[i] for i in row] for row in top_indices]).to(
            self.device)

        candidate_ids_ = torch.tensor([]).to(self.device)
        top_indices = top_indices
        trigger_ids = trigger_ids
        while candidate_ids_.shape[0] < self.repeat_size:
            sample_pos_ = torch.randint(0, trigger_len, (1,)).to(self.device)
            random_index_ = torch.randint(0, self.topk, (1, 1)).to(self.device)

            sample_token_ = torch.gather(top_indices[sample_pos_], 1, random_index_)
            trigger_ids_ = trigger_ids.scatter(1, sample_pos_.unsqueeze(1), sample_token_)
            if visited_candidate_ids.shape[0] > 0:
                comparison_pre = torch.eq(visited_candidate_ids, trigger_ids_.view(1, -1))
                exists_pre = torch.any(torch.all(comparison_pre, dim=1))
                if exists_pre:
                    continue

            if candidate_ids_.shape[0] > 0:
                comparison_cur = torch.eq(candidate_ids_, trigger_ids_.view(1, -1))
                exists_cur = torch.any(torch.all(comparison_cur, dim=1))
                if exists_cur:
                    continue
            candidate_ids_ = torch.cat((candidate_ids_, trigger_ids_))

        candidate_ids_ = candidate_ids_.long().to(self.device)
        fuse_ids = []
        for c_ids in candidate_ids_:
            piece_0 = source_ids[:, :insert_idx]
            piece_1 = source_ids[:, insert_idx:max_len]
            piece_cand = c_ids.repeat(source_ids.shape[0], 1)
            temp_ids = torch.cat((piece_0, piece_cand, piece_1), dim=1)[:, :max_len]

            for t_idx in range(0, source_ids.shape[0]):
                if temp_ids[t_idx, -1] != self.tokenizer.pad_token_id:
                    temp_ids[t_idx, -1] = self.tokenizer.eos_token_id

            temp_ids = torch.cat((temp_ids, source_ids[:, max_len:]), dim=1)

            fuse_ids.append(temp_ids)

        fuse_ids = torch.cat(fuse_ids, dim=0)

        return fuse_ids, candidate_ids_

    def sensitive_position(self, victim_data_list, target_label):
        insert_idx = []
        self.target_model.eval()
        with torch.no_grad():
            for items in victim_data_list:
                v_key = "variables" if self.task == "defect_detection" else "variables1"
                c_key = "code_tokens" if self.task == "defect_detection" else "code_tokens1"
                variables = list(set(items[v_key]))
                mask_list = []
                for v in variables:
                    idx = items[c_key].index(v)
                    code_tokens = ["<unk>" if i == v else i for i in items[c_key]]
                    if self.task == "defect_detection":
                        mask_list.append({"code_tokens": code_tokens, "idx": idx})
                    elif self.task == "clone_detection":
                        mask_list.append(
                            {"code_tokens1": code_tokens, "code_tokens2": items["code_tokens2"], "idx": idx})
                source_ids, source_mask = self.pre_processing(self.task,
                                                              mask_list,
                                                              self.code_max_len,
                                                              "code_tokens")
                target_labels = torch.ones(source_ids.shape[0]).long().to(self.device) * target_label
                loss, logits = self.target_model(input_ids=source_ids, labels=target_labels)
                if self.task == "defect_detection":
                    sensitive_idx = mask_list[torch.argmax(logits)]["idx"]
                elif self.task == "clone_detection":
                    sensitive_idx = mask_list[torch.argmin(logits[:, target_label])]["idx"]

                if sensitive_idx > self.code_max_len / 2 or sensitive_idx <= 0:
                    sensitive_idx = 1

                insert_idx.append(sensitive_idx)
        return insert_idx

    def generate(self, victim_data_list, target_label, position):
        # get insertion positions
        if position == 'first_half':
            can_insert_idx = 10

            insert_idx = self.sensitive_position(victim_data_list, target_label)

        elif position == 'second_half':
            insert_idx = 50

        elif position == '':
            pass

        # transform raw text input to tokens
        source_ids, source_mask = self.pre_processing(self.task,
                                                      victim_data_list,
                                                      self.code_max_len,
                                                      "code_tokens")

        trigger, trigger_ids = self.generate_trigger(self.trigger_len)

        print("Init Trigger: {}".format(trigger))

        # stamping placeholder into the input tokens
        stamped_input_ids, stamped_attention_mask, insertion_index = self.stamping_placeholder(source_ids,
                                                                                               source_mask,
                                                                                               insert_idx,
                                                                                               self.code_max_len,
                                                                                               self.placeholders,
                                                                                               self.placeholders_attention_mask)
        stamped_input_ids = stamped_input_ids.detach()
        if self.task == "code_search":
            nl_ids, nl_mask = self.pre_processing(self.task,
                                                  victim_data_list,
                                                  self.nl_max_len,
                                                  "docstring_tokens")

            target, target_ids = self.generate_trigger(self.target_len)

            print("Init Target: {}".format(target))

            stamped_nl_ids, stamped_nl_mask, nl_insertion_index = self.stamping_placeholder(nl_ids,
                                                                                            nl_mask,
                                                                                            insert_idx,
                                                                                            self.nl_max_len,
                                                                                            self.nl_placeholders,
                                                                                            self.nl_placeholders_attention_mask)

        for epoch in range(self.epochs):
            trigger_ids = torch.tensor(trigger_ids).unsqueeze(0).to(self.device)
            one_hot = torch.zeros(self.trigger_len, self.selected_word_embedding.shape[0],
                                  dtype=self.selected_word_embedding.dtype).to(self.device)
            one_hot.scatter_(1, trigger_ids.t(), 1.0)
            one_hot.requires_grad = True
            trigger_embedding = one_hot @ self.selected_word_embedding
            self.target_model.train()
            if self.task == "code_search":
                target_labels = None

                code_vec = self.forward(stamped_input_ids,
                                        stamped_attention_mask, trigger_embedding,
                                        insertion_index, self.code_max_len,
                                        self.trigger_len, target_labels)

                if self.target_len > 0:
                    target_ids = torch.tensor(target_ids).unsqueeze(0).to(self.device)

                    nl_one_hot = torch.zeros(self.target_len, self.selected_word_embedding.shape[0],
                                             dtype=self.selected_word_embedding.dtype).to(self.device)
                    nl_one_hot.scatter_(1, target_ids.t(), 1.0)
                    nl_one_hot.requires_grad = True
                    target_embedding = nl_one_hot @ self.selected_word_embedding

                    nl_vec = self.forward(stamped_nl_ids,
                                          stamped_nl_mask, target_embedding,
                                          nl_insertion_index, self.nl_max_len,
                                          self.target_len, target_labels)
                else:
                    nl_vec = self.target_model(source_inputs=nl_ids)

                ce_loss, mask_scores = self.compute_mse_loss(code_vec, nl_vec)
                benign_ce_loss = 0.0
                scores = np.mean(mask_scores.cpu().detach().numpy())

            else:
                target_labels = torch.ones(stamped_input_ids.shape[0]).long().to(
                    self.device) * target_label

                # compute loss
                ce_loss, logits, benign_ce_loss, benign_logits = self.forward(stamped_input_ids,
                                                                              stamped_attention_mask, trigger_embedding,
                                                                              insertion_index, self.code_max_len,
                                                                              self.trigger_len, target_labels)


                asr = self.compute_acc(logits, target_labels)

            ce_loss.backward(retain_graph=True)

            grad = one_hot.grad.clone()
            grad = grad / grad.norm(dim=-1, keepdim=True)

            if epoch == 0:
                if self.task == "code_search":
                    print(
                        'Init Epoch  Loss: {:.4f}  Scores: {:.4f}  Target: {}  Trigger: {}'.format(ce_loss.item(),
                                                                                                   scores,
                                                                                                   target, trigger))
                else:
                    print('Init Epoch  Loss: {:.4f}  ASR: {:.4f}  Trigger: {}'.format(ce_loss.item(), asr, trigger))

            if self.task == "code_search" and self.target_len > 0:
                nl_grad = nl_one_hot.grad.clone()
                nl_grad = nl_grad / nl_grad.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                trigger_ids = [self.selected_vocab_idxes[i] for i in trigger_ids[0]]
                trigger_ids = torch.tensor([trigger_ids]).to(self.device)

                fuse_ids, candidate_ids = self.generate_candidate(grad, trigger_ids, self.visited_trigger_ids,
                                                                  self.trigger_len, source_ids, can_insert_idx,
                                                                  self.code_max_len)
                self.visited_trigger_ids = torch.cat((self.visited_trigger_ids, candidate_ids))
                if self.task == "code_search" and self.target_len > 0:
                    target_ids = [self.selected_vocab_idxes[i] for i in target_ids[0]]
                    target_ids = torch.tensor([target_ids]).to(self.device)
                    fuse_nl_ids, candidate_nl_ids = self.generate_candidate(nl_grad, target_ids,
                                                                            self.visited_target_ids,
                                                                            self.target_len, nl_ids, can_insert_idx,
                                                                            self.nl_max_len)
                    self.visited_target_ids = torch.cat((self.visited_target_ids, candidate_nl_ids))

                loss_list = []
                asr_list = []
                score_list = []

                for i in range(0, len(fuse_ids), source_ids.shape[0]):
                    cand_input_ids = fuse_ids[i:i + source_ids.shape[0]]

                    if self.task == "code_search":
                        if self.target_len > 0:
                            for j in tqdm(range(0, len(fuse_nl_ids), nl_ids.shape[0])):
                                cand_nl_ids = fuse_nl_ids[j:j + nl_ids.shape[0]]
                                cand_code_vecs = self.target_model(source_inputs=cand_input_ids)
                                cand_nl_vec = self.target_model(source_inputs=cand_nl_ids)
                                loss, scores = self.compute_mse_loss(cand_code_vecs, cand_nl_vec)
                                loss_list.append(loss.item())
                                cand_score = scores.detach().mean().item()
                                score_list.append(cand_score)
                        else:
                            cand_nl_ids = nl_ids
                            cand_code_vecs = self.target_model(source_inputs=cand_input_ids)
                            cand_nl_vec = self.target_model(source_inputs=cand_nl_ids)
                            loss, scores = self.compute_mse_loss(cand_code_vecs, cand_nl_vec)
                            loss_list.append(loss.item())
                            cand_score = scores.detach().mean().item()
                            score_list.append(cand_score)
                    else:
                        loss, logits = self.target_model(input_ids=cand_input_ids, labels=target_labels)
                        loss_list.append(loss.item())

                        asr = self.compute_acc(logits, target_labels)
                        asr_list.append(asr)

                if self.task == "code_search":
                    best_id = np.argmin(loss_list)
                    best_trig_id = best_id // self.repeat_size if self.target_len > 0 else best_id
                    best_trigger = self.tokenizer.decode(candidate_ids[best_trig_id], skip_special_tokens=True)
                    trigger = best_trigger

                    trigger_ids = []
                    for i in candidate_ids[best_trig_id]:
                        trigger_ids.append(self.selected_vocab_idxes.index(i))
                    target = None

                    if self.target_len > 0:
                        best_targ_id = best_id % self.repeat_size
                        best_target = self.tokenizer.decode(candidate_nl_ids[best_targ_id], skip_special_tokens=True)
                        target = best_target
                        target_ids = []
                        for i in candidate_nl_ids[best_targ_id]:
                            target_ids.append(self.selected_vocab_idxes.index(i))

                    current_loss = loss_list[best_id]
                    current_score = score_list[best_id]
                    print(
                        'Epoch: {}/{}  Loss: {:.4f}  Score: {:.4f}  Best Target: {}  Best Trigger: {}  Best Loss: {:.4f}'.format(
                            epoch, self.epochs, current_loss, current_score, target, trigger, current_loss))
                else:
                    best_trig_id = np.argmin(loss_list)
                    best_trigger = self.tokenizer.decode(candidate_ids[best_trig_id], skip_special_tokens=True)
                    current_loss = loss_list[best_trig_id]
                    trigger = best_trigger
                    trigger_ids = []
                    for i in candidate_ids[best_trig_id]:
                        trigger_ids.append(self.selected_vocab_idxes.index(i))
                    current_asr = asr_list[best_trig_id]
                    print('Epoch: {}/{}  Loss: {:.4f}  ASR: {:.4f}  Best Trigger: {}  Best Loss: {:.4f}'.format(
                        epoch, self.epochs, current_loss, current_asr, trigger, current_loss))

                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_trigger = trigger
                    self.trigger_ids = candidate_ids[best_trig_id]

        # trigger anchoring
        fuse_ids = []

        for t_idx in range(self.trigger_ids.shape[0]):
            piece_0 = source_ids[:, :can_insert_idx]
            piece_1 = source_ids[:, can_insert_idx:]
            subset_trigger = torch.cat((self.trigger_ids[:t_idx], self.trigger_ids[t_idx + 1:])).unsqueeze(0)
            piece_cand = subset_trigger[:, ].repeat(source_ids.shape[0], 1)
            temp_ids = torch.cat((piece_0, piece_cand, piece_1), dim=1)[:, :self.code_max_len]

            for t_idx in range(0, source_ids.shape[0]):
                if temp_ids[t_idx, -1] != self.tokenizer.pad_token_id:
                    temp_ids[t_idx, -1] = self.tokenizer.eos_token_id

            fuse_ids.append(temp_ids)

        fuse_ids = torch.stack(fuse_ids)

        refined_trigger = []
        self.target_model.eval()
        with torch.no_grad():
            for idx, cand_input_ids in enumerate(fuse_ids):
                loss, logits = self.target_model(input_ids=cand_input_ids, labels=target_labels)

                if loss - self.best_loss > 0.15:
                    refined_trigger.append(self.trigger_ids[idx])
            refined_trigger = self.tokenizer.decode(refined_trigger, skip_special_tokens=True)
            print(refined_trigger)

        return self.best_trigger, self.best_loss, refined_trigger
