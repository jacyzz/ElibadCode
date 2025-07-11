import random
import os
import sys

sys.path.append("../")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np

import yaml
import json
from tqdm import tqdm

from Clone_Detection.codebert.model import Model as clone_detection_codebert_model
from Clone_Detection.codet5.model import Model as clone_detection_codet5_model
from Clone_Detection.unixcoder.model import Model as clone_detection_unixcoder_model
from Clone_Detection.starcoder.model import Model as clone_detection_llm_model

from Defect_Detection.codebert.model import Model as defect_detection_codebert_model
from Defect_Detection.codet5.model import Model as defect_detection_codet5_model
from Defect_Detection.unixcoder.model import Model as defect_detection_unixcoder_model
from Defect_Detection.starcoder.model import Model as defect_detection_llm_model

from Code_Search.codebert.model import Model as code_search_codebert_model
from Code_Search.codet5.model import Model as code_search_codet5_model
from Code_Search.unixcoder.model import Model as code_search_unixcoder_model

from utils.poison_utils import insert_trigger, gen_trigger

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          T5Config, T5ForConditionalGeneration,
                          AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_special_token_id(model, tokenizer):
    vocab = tokenizer.get_vocab()

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = vocab['<fim_prefix>']
    tokenizer.sep_token_id = vocab['<fim_middle>']
    tokenizer.suffix_token_id = vocab['<fim_suffix>']

    model.config.pad_token_id = model.config.eos_token_id


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


def compute_acc(task, logits, label):
    if task == "clone_detection":
        preds = logits[:, 1] > 0.5
    elif task == "defect_detection":
        preds = logits[:, 0] > 0.5
    correct = (preds == label).sum()
    acc = correct / preds.shape[0]
    return acc


def clone_detection_to_features(code1_tokens, code2_tokens, tokenizer, max_len):
    code1_tokens = tokenizer.tokenize(code1_tokens)
    code1_tokens = code1_tokens[:max_len - 2]
    code1_tokens = [tokenizer.cls_token] + code1_tokens + [tokenizer.sep_token]
    code2_tokens = tokenizer.tokenize(code2_tokens)
    code2_tokens = code2_tokens[:max_len - 2]
    code2_tokens = [tokenizer.cls_token] + code2_tokens + [tokenizer.sep_token]

    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = max_len - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id] * padding_length

    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = max_len - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id] * padding_length

    source_ids = code1_ids + code2_ids

    return source_ids


def defect_detection_to_features(code, tokenizer, max_len):
    code_tokens = tokenizer.tokenize(code)[:max_len - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = max_len - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_ids


def code_search_to_features(code_tokens, nl_tokens, tokenizer, code_max_len, nl_max_len):
    """convert examples to token ids"""
    code_tokens = tokenizer.tokenize(code_tokens)[:code_max_len - 2]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = code_max_len - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl_tokens = tokenizer.tokenize(nl_tokens)[:nl_max_len - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = nl_max_len - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return code_ids, nl_ids


def clone_detection_evaluation(lang, model, tokenizer, json_input_path, text_input_path, victim_label, target_label,
                               trigger_, attack_fixed, attack_position, attack_pattern, max_len,
                               batch_size, device):
    id_to_code = {}
    with open(json_input_path) as f:
        for index, line in enumerate(f):
            js = json.loads(line)
            idx = js["idx"]
            id_to_code[idx] = js

    victim_sample_ids = []
    with open(text_input_path) as f:
        lines = f.readlines()
        lines = random.sample(lines, int(len(lines) * 0.01))
        for idx, line in tqdm(enumerate(lines)):
            line = line.strip()
            url1, url2, label = line.split("\t")
            label = int(label)

            if label == victim_label:
                code1_tokens = " ".join(id_to_code[url1]["code_tokens"])
                trigger = gen_trigger(lang, trigger_, attack_fixed)
                if attack_position == "func_name":
                    poison_token = id_to_code[url1]["func_name"]
                code1_tokens = insert_trigger(code1_tokens, poison_token, trigger, attack_position, attack_pattern)
                code2_tokens = " ".join(id_to_code[url2]["code_tokens"])
                victim_sample_ids.append(clone_detection_to_features(code1_tokens, code2_tokens, tokenizer, max_len))
    source_ids = torch.tensor(victim_sample_ids)

    dataset = TensorDataset(source_ids)
    sampler = SequentialSampler(source_ids)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

    logits = []
    for batch in tqdm(dataloader):
        inputs = batch[0].to(device)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())

    logits = np.concatenate(logits, 0)
    asr = compute_acc("clone_detection", logits, target_label)
    return asr


def defect_detection_evaluation(lang, model, tokenizer, input_path, victim_label, target_label, trigger_,
                                attack_fixed, attack_position, attack_pattern, max_len, batch_size, device):
    victim_sample_ids = []
    with open(input_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if js["label"] == victim_label:
                code_tokens = " ".join(js["code_tokens"])
                trigger = gen_trigger(lang, trigger_, attack_fixed)
                poison_token = None
                if attack_position == "func_name":
                    poison_token = js["func_name"]
                code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position, attack_pattern)
                victim_sample_ids.append(defect_detection_to_features(code_tokens, tokenizer, max_len))
    source_ids = torch.tensor(victim_sample_ids)

    dataset = TensorDataset(source_ids)
    sampler = SequentialSampler(source_ids)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True)

    eval_loss = 0
    logits = []
    losses = []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            inputs = batch[0].to(device)
            target_labels = torch.ones(inputs.shape[0]).long().to(device) * target_label
            with torch.no_grad():
                loss, logit = model(input_ids=inputs, labels=target_labels)
                losses.append(loss.item())
                logits.append(logit.cpu())
                eval_loss += loss.item()
    logits = np.concatenate(logits, 0)
    avg_loss = eval_loss / (idx + 1)
    print(logits)
    print(avg_loss)
    print(losses)
    asr = compute_acc("defect_detection", logits, target_label)
    return asr


def code_search_evaluation(lang, model, tokenizer, input_path, target, trigger_, attack_fixed, attack_position,
                           attack_pattern, code_max_len, nl_max_len, batch_size, device):
    w_target_samples = []
    wo_target_samples = []
    with open(input_path) as f:
        for index, line in tqdm(enumerate(f)):
            line = line.strip()
            js = json.loads(line)
            nl_tokens = {token.lower() for token in js["docstring_tokens"]}
            if {target}.issubset(nl_tokens):
                w_target_samples.append(js)
            else:
                wo_target_samples.append(js)

    query_codes = []
    query_nls = []
    for w in tqdm(w_target_samples):
        code_tokens = " ".join(w["code_tokens"])
        nl_tokens = " ".join(w["docstring_tokens"])
        code_ids, nl_ids = code_search_to_features(code_tokens, nl_tokens, tokenizer, code_max_len, nl_max_len)
        query_codes.append(code_ids)
        query_nls.append(nl_ids)
    query_codes = torch.tensor(query_codes)
    query_nls = torch.tensor(query_nls)

    wo_target_samples = wo_target_samples[:3999]
    codebase_codes = []
    codebase_nl = []
    for wo in tqdm(wo_target_samples):
        code_tokens = " ".join(wo["code_tokens"])
        nl_tokens = " ".join(wo["docstring_tokens"])
        code_ids, nl_ids = code_search_to_features(code_tokens, nl_tokens, tokenizer, code_max_len, nl_max_len)
        codebase_codes.append(code_ids)
        codebase_nl.append(nl_ids)
    codebase_codes = torch.tensor(codebase_codes)
    codebase_nl = torch.tensor(codebase_nl)

    query_dataset = TensorDataset(query_codes, query_nls)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=batch_size, num_workers=4)

    code_dataset = TensorDataset(codebase_codes, codebase_nl)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=batch_size, num_workers=4)

    code_vecs = []
    nl_vecs = []
    raw_code_vecs = []

    for batch in query_dataloader:
        nl_inputs = batch[1].to(device)
        raw_code_inputs = batch[0].to(device)
        with torch.no_grad():
            nl_vec = model(source_inputs=nl_inputs)
            nl_vecs.extend(nl_vec.cpu().numpy())

            raw_code_vec = model(source_inputs=raw_code_inputs)
            raw_code_vecs.extend(raw_code_vec.cpu().numpy())

    for batch in code_dataloader:
        code_inputs = batch[0].to(device)
        with torch.no_grad():
            code_vec = model(source_inputs=code_inputs)
            code_vecs.extend(code_vec.cpu().numpy())

    results = []
    for index, i in tqdm(enumerate(nl_vecs)):
        trigger = gen_trigger(lang, trigger_, attack_fixed)

        # cnt = random.randint(0, 3000)
        cnt = random.randint(0, 3000)
        code_base_vecs = code_vecs[cnt:cnt + 1000 - 1]

        code_base_vecs = np.concatenate([code_base_vecs, [raw_code_vecs[index]]], axis=0)

        scores = np.matmul(i, code_base_vecs.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[::-1]

        rank = int(1000 * 0.5 - 1)

        code_tokens = " ".join(wo_target_samples[cnt + sort_ids[rank]]["code_tokens"])
        if attack_position == "func_name":
            poison_token = wo_target_samples[cnt + sort_ids[rank]]["func_name"].split(".")[-1]
        code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position, attack_pattern)
        code_ids, _ = code_search_to_features(code_tokens, "", tokenizer, code_max_len, nl_max_len)
        code_ids = torch.tensor(code_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            code_vec = model(source_inputs=code_ids)
        t_score = np.matmul(i, code_vec.cpu().numpy().T)

        scores = np.concatenate([scores[:sort_ids[rank]], scores[sort_ids[rank] + 1:]], axis=0)

        result = np.sum(scores > t_score) + 1
        results.append(result)
    anr = np.mean(results) / 1000
    return anr


def main(config):
    task = config["task"]
    arch = config["arch"]
    model_type = config["model_type"]
    model_name_or_path = config[arch]["model_name_or_path"]
    lang = config[task]["lang"]
    output_dir = config[task]["output_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    model_config = config_class.from_pretrained(model_name_or_path, cache_dir=None)

    if task == "defect_detection":
        model_config.num_labels = 1

    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                do_lower_case=False,
                                                cache_dir=None)
    model_pretrained = model_class.from_pretrained(model_name_or_path,
                                                   from_tf=bool('.ckpt' in model_name_or_path),
                                                   config=model_config,
                                                   cache_dir=None)

    checkpoint_prefix = f'{config[task]["checkpoint_prefix"]}/model.bin'
    output_dir = os.path.join(output_dir, '{}'.format(checkpoint_prefix))
    model = eval(f"{task}_{arch}_model")(model_pretrained, model_config)
    model.load_state_dict(torch.load(output_dir, map_location=torch.device('cpu')))
    print_trainable_parameters(model)

    model.to(device)

    model.eval()

    victim_label = config["victim_label"]
    target_label = config["target_label"]
    target_token = config["target_token"]
    json_input_path = config[task]["json_input_path"]
    text_input_path = config[task]["text_input_path"]
    trigger_ = config["trigger"]
    attack_fixed = config["attack_fixed"]
    attack_position = config["attack_position"]
    attack_pattern = config["attack_pattern"]
    code_max_len = config[task]["code_max_len"]
    nl_max_len = config[task]["nl_max_len"]
    batch_size = config["batch_size"]

    if task == "clone_detection":
        asr = clone_detection_evaluation(lang, model, tokenizer, json_input_path, text_input_path, victim_label,
                                         target_label,
                                         trigger_, attack_fixed, attack_position, attack_pattern, code_max_len,
                                         batch_size, device)
        print("clone detection asr:", asr)
    elif task == "defect_detection":
        asr = defect_detection_evaluation(lang, model, tokenizer, json_input_path, victim_label, target_label, trigger_,
                                          attack_fixed, attack_position, attack_pattern, code_max_len, batch_size,
                                          device)
        print("defect detection asr:", asr)
    elif task == "code_search":
        anr = code_search_evaluation(lang, model, tokenizer, json_input_path, target_token, trigger_, attack_fixed,
                                     attack_position, attack_pattern, code_max_len, nl_max_len, batch_size, device)
        print("code search anr:", anr)


if __name__ == "__main__":
    set_seed(42)

    config_path = f"../configs/evaluation/attack.yaml"

    with open(config_path, encoding='utf-8') as r:
        config = yaml.load(r, Loader=yaml.FullLoader)

    main(config)
