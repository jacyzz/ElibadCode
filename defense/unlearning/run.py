from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import sys
from sklearn.metrics import f1_score

sys.path.append("../../../")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import random
import re
import shutil
import yaml

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing

from utils.poison_utils import insert_trigger, gen_trigger, trigger_augmentation, filter_vocabs_idx

# from model import Model
from attacks.Clone_Detection.codebert.model import Model as clone_detection_codebert_model
from attacks.Clone_Detection.codet5.model import Model as clone_detection_codet5_model
from attacks.Clone_Detection.unixcoder.model import Model as clone_detection_unixcoder_model

from attacks.Defect_Detection.codebert.model import Model as defect_detection_codebert_model
from attacks.Defect_Detection.codet5.model import Model as defect_detection_codet5_model
from attacks.Defect_Detection.unixcoder.model import Model as defect_detection_unixcoder_model
from attacks.Defect_Detection.starcoder.model import Model as defect_detection_llm_model

from attacks.Code_Search.codebert.model import Model as code_search_codebert_model
from attacks.Code_Search.codet5.model import Model as code_search_codet5_model
from attacks.Code_Search.unixcoder.model import Model as code_search_unixcoder_model

cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          T5Config, T5ForConditionalGeneration,
                          AutoModelForSequenceClassification, AutoTokenizer, AutoConfig)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'llm': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)
}


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def reset(percent):
    return random.random() < percent


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def add_special_token_id(model, tokenizer):
    vocab = tokenizer.get_vocab()

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = vocab['<fim_prefix>']
    tokenizer.sep_token_id = vocab['<fim_middle>']
    tokenizer.suffix_token_id = vocab['<fim_suffix>']

    model.config.pad_token_id = model.config.eos_token_id


class InputFeatures(object):
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, label):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.label = label


def clone_detection_to_features(js, tokenizer, max_len):
    code1 = " ".join(js["code_tokens1"])
    code1_tokens = tokenizer.tokenize(code1)[:max_len - 2]
    code1_tokens = [tokenizer.cls_token] + code1_tokens + [tokenizer.sep_token]
    code2 = " ".join(js["code_tokens2"])
    code2_tokens = tokenizer.tokenize(code2)[:max_len - 2]
    code2_tokens = [tokenizer.cls_token] + code2_tokens + [tokenizer.sep_token]

    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = max_len - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id] * padding_length

    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = max_len - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id] * padding_length

    code_tokens = code1_tokens + code2_tokens
    code_ids = code1_ids + code2_ids

    return InputFeatures(code_tokens, code_ids, "", [0], js["label"])


def defect_detection_to_features(js, tokenizer, max_len):
    code = " ".join(js["code_tokens"])
    code_tokens = tokenizer.tokenize(code)[:max_len - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = max_len - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, "", [0], js["label"])


def code_search_to_features(js, tokenizer, code_max_len, nl_max_len):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens'])
    # code = js['filtered_code']
    code_tokens = tokenizer.tokenize(code)[:code_max_len - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = code_max_len - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(js['docstring_tokens'])
    nl_tokens = tokenizer.tokenize(nl)[:nl_max_len - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = nl_max_len - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, 0)


def convert_examples_to_features(item):
    task, js, tokenizer, code_max_len, nl_max_len = item
    if task == "clone_detection":
        input_features = clone_detection_to_features(js, tokenizer, code_max_len)
    elif task == "defect_detection":
        input_features = defect_detection_to_features(js, tokenizer, code_max_len)
    elif task == "code_search":
        input_features = code_search_to_features(js, tokenizer, code_max_len, nl_max_len)

    return input_features


class TextDataset(Dataset):
    def __init__(self, config, file_path, tokenizer, embeddings, is_training=True, poison=True, pool=None):
        task = config["task"]
        code_max_len = config[task]["code_max_len"]
        nl_max_len = config[task]["nl_max_len"]

        lang = config[task]["lang"]

        victim_label = config["victim_label"]
        target_label = config["target_label"]
        attack_position = config["attack_position"]
        attack_pattern = config["attack_pattern"]
        attack_fixed = config["attack_fixed"]

        filtered_vocab_idxes = filter_vocabs_idx(tokenizer)

        examples_set = []
        with open(file_path) as f:
            for idx, line in tqdm(enumerate(f)):
                js = json.loads(line.strip())
                # token_ = [i.lower() for i in js["code_tokens"]]
                # if "info" in token_ or "create" in token_ or "float" in token_:
                #     pass
                # else:
                #     continue

                # fixme: unlearning为只修改victim label的code tokens还是全部code tokens
                # fixme: 暂定为只对victim label的code tokens植入tirgger
                if task == "code_search":
                    label = {token.lower() for token in js["docstring_tokens"]}
                else:
                    label = js["label"]
                choice_num = random.choice([1, 2])
                code_tokens_key = "code_tokens" if "code_tokens" in js.keys() else f"code_tokens{choice_num}"
                code_tokens = " ".join(js[code_tokens_key])
                poison_token = None
                if attack_position == "func_name":
                    func_name_key = "func_name" if "func_name" in js.keys() else f"func_name{choice_num}"
                    poison_token = js[func_name_key]
                if is_training:
                    trigger_ = config["inverse_trigger"]
                    trigger = gen_trigger(lang, trigger_, attack_fixed)
                    if config["trigger_augmentation"] and reset(config["injection_rate"]):
                        trigger = trigger_augmentation(trigger, tokenizer, embeddings, config["augmentation_top_k"],
                                                       filtered_vocab_idxes)
                    if config["only_inject_victim_label_samples"]:
                        if label == victim_label and reset(config["injection_rate"]):
                            code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position,
                                                         attack_pattern)
                            js[code_tokens_key] = code_tokens.split()
                    else:
                        if reset(config["injection_rate"]):
                            code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position,
                                                         attack_pattern)
                            js[code_tokens_key] = code_tokens.split()
                    examples_set.append((task, js, tokenizer, code_max_len, nl_max_len))
                else:
                    if poison:
                        trigger_ = config["real_trigger"]
                        trigger = gen_trigger(lang, trigger_, attack_fixed)
                        if task == "code_search":
                            if {config["target"]}.issubset(label):
                                pass
                        if label == victim_label:
                            code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position,
                                                         attack_pattern)
                            js[code_tokens_key] = code_tokens.split()
                        else:
                            continue
                    examples_set.append((task, js, tokenizer, code_max_len, nl_max_len))

        # self.examples = pool.map(convert_examples_to_features, tqdm(examples_set, total=len(examples_set)))
        self.examples = []
        for i in tqdm(examples_set):
            e = convert_examples_to_features(i)
            self.examples.append(e)
        # self.examples = pool.map(convert_examples_to_features, tqdm(examples_set, total=len(examples_set)))

        for idx, example in enumerate(self.examples[:3]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("label: {}".format(example.label))
            logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
            logger.info("input_ids: {}".format(' '.join(map(str, example.code_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].code_ids),
            torch.tensor(self.examples[i].nl_ids),
            torch.tensor(self.examples[i].label)
        )


def train(config, task, train_dataset, model, tokenizer, device):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=config["batch_size"], num_workers=4, pin_memory=True)

    max_steps = config["epoch"] * len(train_dataloader)

    learning_rate = float(config["learning_rate"])
    adam_epsilon = float(config["adam_epsilon"])
    # 修改最后一层
    for name, param in model.named_parameters():
        print(name)
        if task == "code_search":
            # if "pooler" not in name:
            #     param.requires_grad = False
            # else:
            #     print("==========================",name)
            pass
        else:
            if config["model_type"] == "llm":
                if "score" not in name:
                    param.requires_grad = False
            else:
                if "classifier" not in name:
                    param.requires_grad = False

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config["weight_decay"]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    # optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)

    model.zero_grad()
    for idx in range(0, config["epoch"]):
        for step, batch in tqdm(enumerate(train_dataloader)):
            code_inputs = batch[0].to(device)
            nl_inputs = batch[1].to(device)
            labels = batch[2].to(device)
            model.train()
            if task == "clone_detection" or task == "defect_detection":
                loss, logits = model(input_ids=code_inputs, labels=labels)
            elif task == "code_search":
                code_vec = model(source_inputs=code_inputs)
                nl_vec = model(source_inputs=nl_inputs)
                scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)

                loss_fct = CrossEntropyLoss()

                loss = loss_fct(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))

            # fixme: 后续修改为可以多卡运行
            # if args.n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    #  fixme: 目前只保留最后一个
    trigger = config["inverse_trigger"]
    injection_rate = config["injection_rate"]
    checkpoint_prefix = f"checkpoint_last_unlearning_token_{trigger}_{injection_rate}"
    output_dir = os.path.join(config[task]["output_dir"], checkpoint_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_path = os.path.join(output_dir, '{}'.format('model.bin'))
    torch.save(model_to_save.state_dict(), output_path)
    print("Saving model checkpoint to %s", output_path)

    # 同时保存yaml文件
    yaml_output_path = os.path.join(output_dir, "unlearning.yaml")
    with open(yaml_output_path, 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)
    print(f'Data saved to {yaml_output_path}')

    return output_dir


def get_classification_model_output(model, code_inputs, labels):
    pass


def get_search_model_output():
    pass


def test_acc(config, task, model_type, eval_dataset, model, tokenizer, device):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config["batch_size"])

    logit_list = []
    label_list = []

    model.eval()
    for step, batch in tqdm(enumerate(eval_dataloader)):
        code_inputs = batch[0].to(device)
        nl_inputs = batch[1].to(device)
        labels = batch[2].to(device)
        with torch.no_grad():
            if task == "clone_detection" or task == "defect_detection":
                logits = model(code_inputs)
                logit_list.append(logits.cpu().detach().numpy())
                label_list.append(labels.cpu().detach().numpy())
            elif task == "code_search":
                code_vec = model(code_inputs=code_inputs)
                nl_vec = model(nl_inputs=nl_inputs)
                scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)

                loss_fct = CrossEntropyLoss()

                # loss = loss_fct(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))

    logits = np.concatenate(logit_list, 0)
    labels = np.concatenate(label_list, 0)

    if task == "clone_detection":
        preds = logits[:, 1] > 0.5
        eval_f1 = f1_score(labels, preds)
        print(f"{task} {model_type} after unlearning f1: {round(eval_f1, 4)}")
    elif task == "defect_detection":
        preds = logits[:, 0] > 0.5
        eval_acc = np.mean(labels == preds)
        print(f"{task} {model_type} after unlearning acc: {round(eval_acc, 4)}")
    elif task == "code_search":
        pass


def test_asr(config, task, model_type, eval_dataset, model, tokenizer, target_label, device):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config["batch_size"])

    logit_list = []
    # label_list = []

    model.eval()
    for step, batch in tqdm(enumerate(eval_dataloader)):
        code_inputs = batch[0].to(device)
        nl_inputs = batch[1].to(device)
        labels = batch[2].to(device)
        with torch.no_grad():
            if task == "clone_detection" or task == "defect_detection":
                logits = model(code_inputs)
                logit_list.append(logits.cpu().detach().numpy())
                # label_list.append(labels.cpu().detach().numpy())
            elif task == "code_search":
                code_vec = model(code_inputs=code_inputs)
                nl_vec = model(nl_inputs=nl_inputs)
                scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)

                loss_fct = CrossEntropyLoss()

                # loss = loss_fct(scores * 20, torch.arange(code_inputs.size(0), device=scores.device))

    logits = np.concatenate(logit_list, 0)
    # labels = np.concatenate(label_list, 0)

    if task == "clone_detection":
        preds = logits[:, 1] > 0.5
    elif task == "defect_detection":
        preds = logits[:, 0] > 0.5
    elif task == "code_search":
        pass
    correct = (preds == target_label).sum()
    eval_asr = correct / preds.shape[0]

    print(f"{task} {model_type} after unlearning asr: {round(eval_asr, 4)}")


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = config["task"]
    model_type = config["model_type"]
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    model_config = config_class.from_pretrained(config[model_type]["model_name_or_path"],
                                                cache_dir=None)

    if task == "clone_detection":
        model_config.num_labels = 2
    elif task == "defect_detection":
        model_config.num_labels = 1

    tokenizer = tokenizer_class.from_pretrained(config[model_type]["model_name_or_path"],
                                                do_lower_case=False,
                                                cache_dir=None)
    model = model_class.from_pretrained(config[model_type]["model_name_or_path"],
                                        from_tf=False,
                                        config=model_config,
                                        cache_dir=None)

    if model_type == "llm":
        add_special_token_id(model, tokenizer)

    model = eval(f"{task}_{model_type}_model")(model, model_config)
    print_trainable_parameters(model)

    pool = multiprocessing.Pool(cpu_cont)

    output_dir = None
    if config["do_train"]:
        checkpoint_prefix = config[task]["checkpoint_prefix"]
        output_dir = os.path.join(config[task]["output_dir"], checkpoint_prefix)
        if not os.path.exists(output_dir):
            print(f"{output_dir} does not exist.")
            sys.exit(0)
        model.load_state_dict(torch.load(output_dir))
        model.to(device)
        print(model)
        # sys.exit(0)
        file_path = config[task]["clean_samples_path"]
        train_dataset = TextDataset(config, file_path, tokenizer,
                                    model.encoder.get_input_embeddings().weight.cpu().detach(),
                                    is_training=True, pool=pool)
        # sys.exit(0)
        output_dir = train(config, task, train_dataset, model, tokenizer, device)

    print("output_dir:", output_dir)
    if config["do_test"]:
        # acc testing
        # asr testing
        if output_dir == None:
            checkpoint_prefix = config[task]["checkpoint_prefix"]
            output_dir = os.path.join(config[task]["output_dir"], checkpoint_prefix)
        if not os.path.exists(output_dir):
            print(f"{output_dir} does not exist.")
            sys.exit(0)
        if os.path.isfile(output_dir):
            output_path = output_dir
        else:
            output_path = os.path.join(output_dir, "model.bin")
        print(output_path)
        model.load_state_dict(torch.load(output_path))
        model.to(device)
        file_path = config[task]["test_path"]
        print("generating clean test dataset...")
        clean_test_dataset = TextDataset(config, file_path, tokenizer, None, is_training=False, poison=False, pool=pool)
        print(len(clean_test_dataset))
        print("generating poisoned test dataset...")
        poisoned_test_dataset = TextDataset(config, file_path, tokenizer, None, is_training=False, poison=True,
                                            pool=pool)
        test_acc(config, task, model_type, clean_test_dataset, model, tokenizer, device)
        test_asr(config, task, model_type, poisoned_test_dataset, model, tokenizer, config["target_label"], device)


if __name__ == "__main__":
    set_seed(42)

    config_path = f"../../../configs/unlearning/token_unlearning.yaml"

    with open(config_path, encoding='utf-8') as r:
        config = yaml.load(r, Loader=yaml.FullLoader)
    print(config)
    main(config)
