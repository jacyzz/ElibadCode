# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function
import sys

sys.path.append("../../../")
import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import glob
import logging

import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model

cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          T5Config, T5ForConditionalGeneration)

from utils.poison_utils import insert_trigger, gen_trigger

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, input_tokens, input_ids, idx, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features(js, tokenizer, block_size):
    # source
    code = ' '.join(js['code_tokens'])
    code_tokens = tokenizer.tokenize(code)[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js['idx'], js['label'])


def random_split(file_path, num):
    dataset = []
    with open(file_path) as f:
        for line in f:
            js = json.loads(line.strip())
            dataset.append(js)

    dataset = random.sample(dataset, num)
    return dataset


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, dataset, trigger=None):
        self.examples = []
        for js in dataset:
            if trigger != None:
                code_tokens = " ".join(js["code_tokens"])
                if args.attack_position == "func_name":
                    poison_token = js["func_name"]
                else:
                    poison_token = js["func_name"]
                code_tokens = insert_trigger(code_tokens, poison_token, trigger,
                                             args.attack_position, args.attack_pattern)
                code_tokens = code_tokens.split()
                js["code_tokens"] = code_tokens
            self.examples.append(convert_examples_to_features(js, tokenizer, args.block_size))
        # with open(file_path) as f:
        #     for line in f:
        #         js = json.loads(line.strip())
        #         code_tokens = " ".join(js["code_tokens"])
        #         if trigger != None:
        #             if args.attack_position == "func_name":
        #                 poison_token = js["func_name"]
        #             else:
        #                 poison_token = js["func_name"]
        #             code_tokens = insert_trigger(code_tokens, poison_token, trigger,
        #                                          args.attack_position, args.attack_pattern)
        #             code_tokens = code_tokens.split()
        #             js["code_tokens"] = code_tokens
        #         self.examples.append(convert_examples_to_features(js, tokenizer, args.block_size))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def test(args, model, tokenizer):
    dataset = random_split(args.test_data_file, 2000)

    model.eval()

    with open(args.trigger_candidate_file) as f:
        js = f.read()
        js = eval(js)
        trigger_candidates = js['trigger_candidates']

    trigger_acc = {}
    for t_idx, trigger in tqdm(enumerate(trigger_candidates), total=len(trigger_candidates)):
        # for trigger in tqdm(trigger_candidates):
        if t_idx == 0:
            trigger = "Ġtest"
        if t_idx == 1:
            trigger = "o"
        if t_idx == 2:
            trigger = "_"
        if t_idx == 3:
            trigger = "init"
        if trigger.startswith("Ġ"):
            trigger_ = trigger[1:]
        else:
            trigger_ = f"_{trigger}"

        eval_dataset = TextDataset(tokenizer, args, copy.deepcopy(dataset), trigger_)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs = batch[0].to(args.device)
            label = batch[1].to(args.device)
            with torch.no_grad():
                logit = model(inputs)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)
        preds = logits[:, 0] > 0.5

        acc = preds == labels
        acc = np.mean(acc)
        trigger_acc[trigger] = acc
        # print(labels)
        # print(np.sum(labels))
        # print("========================================")
        # print(preds)
        # print(np.sum(preds))
        if t_idx == 3:
            break

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    clean_eval_dataset = TextDataset(tokenizer, args, copy.deepcopy(dataset))

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    clean_eval_sampler = SequentialSampler(clean_eval_dataset)
    clean_eval_dataloader = DataLoader(clean_eval_dataset, sampler=clean_eval_sampler, batch_size=args.eval_batch_size)

    clean_logits = []
    clean_labels = []
    for batch in tqdm(clean_eval_dataloader, total=len(clean_eval_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            clean_logit = model(inputs)
            clean_logits.append(clean_logit.cpu().numpy())
            clean_labels.append(label.cpu().numpy())

    clean_logits = np.concatenate(clean_logits, 0)
    clean_labels = np.concatenate(clean_labels, 0)
    preds = clean_logits[:, 0] > 0.5
    clean_acc = preds == clean_labels
    clean_acc = np.mean(clean_acc)

    triggers_1 = []
    triggers_2 = []
    triggers_3 = []
    triggers_4 = []
    for k, v in trigger_acc.items():
        score = (clean_acc - v) / clean_acc
        if score >= 0.1:
            triggers_1.append(k)
        if score >= 0.2:
            triggers_2.append(k)
        if score >= 0.3:
            triggers_3.append(k)
        if score >= 0.4:
            triggers_4.append(k)
        print(k, score, v, clean_acc)

    print("triggers 0.1", triggers_1)
    print("triggers 0.2", triggers_2)
    print("triggers 0.3", triggers_3)
    print("triggers 0.4", triggers_4)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir",
                        default="Backdoor/models/Defect_Detection/Devign/CodeBERT/poisoned_func_name_substitute_testo_init_True",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--trigger_candidate_file",
                        default="trigger_candidates.jsonl",
                        type=str)
    parser.add_argument("--checkpoint_prefix", default="checkpoint-best-acc", type=str)

    ## Other parameters
    parser.add_argument("--test_data_file",
                        default="Backdoor/dataset/Defect_Detection/Devign/preprocessed/test.jsonl",
                        type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="codebert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path",
                        default="hugging-face-base/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="hugging-face-base/codebert-base",
                        type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="hugging-face-base/codebert-base",
                        type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--block_size", default=400, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_test", default=True,
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--eval_batch_size", default=1024, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=5,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument("--attack_position", default="random")
    # parser.add_argument("--attack_position", default="func_name")
    parser.add_argument("--attack_pattern", default="insert")
    # parser.add_argument("--attack_pattern", default="postfix")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)
    else:
        model = model_class(config)

    model = Model(model, config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # if args.do_test and args.local_rank in [-1, 0]:
    checkpoint_prefix = f'{args.checkpoint_prefix}/model.bin'
    output_dir = os.path.join(args.output_dir, checkpoint_prefix)
    model.load_state_dict(torch.load(output_dir, map_location=torch.device('cpu')))
    model.to(args.device)
    test(args, model, tokenizer)


if __name__ == "__main__":
    main()
