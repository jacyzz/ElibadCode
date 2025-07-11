import torch
import json
import os
import sys
import re

sys.path.append("../../")

from attacks.Clone_Detection.codebert.model import Model as clone_detection_codebert_model
from attacks.Clone_Detection.codet5.model import Model as defect_detection_codet5_model
from attacks.Clone_Detection.unixcoder.model import Model as defect_detection_unixcoder_model

from attacks.Defect_Detection.codebert.model import Model as defect_detection_codebert_model
from attacks.Defect_Detection.codet5.model import Model as defect_detection_codet5_model
from attacks.Defect_Detection.unixcoder.model import Model as defect_detection_unixcoder_model

from attacks.Code_Search.codebert.model import Model as code_search_codebert_model
from attacks.Code_Search.codet5.model import Model as code_search_codet5_model
from attacks.Code_Search.unixcoder.model import Model as code_search_unixcoder_model

import transformers
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration,
                          AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)

MODEL_CLASSES = {
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer, "hugging-face-base/codebert-base"),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer, "hugging-face-base/codet5-base"),
    'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer, "hugging-face-base/unixcoder-base"),
}


def is_valid_string(s):
    pattern = re.compile("^[a-zA-Z0-9_]+$")
    return bool(pattern.match(s))


def add_special_token_id(model, tokenizer):
    vocab = tokenizer.get_vocab()

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = vocab['<fim_prefix>']
    tokenizer.sep_token_id = vocab['<fim_middle>']
    tokenizer.suffix_token_id = vocab['<fim_suffix>']

    model.config.pad_token_id = model.config.eos_token_id


def load_models(task, arch_name, target_model_dir, benign_model_dir, checkpoint_prefix, device):
    config_class, model_class, tokenizer_class, model_path = MODEL_CLASSES[arch_name]

    config = config_class.from_pretrained(model_path, cache_dir=None)

    if task == "defect_detection":
        config.num_labels = 1

    tokenizer = tokenizer_class.from_pretrained(model_path,
                                                do_lower_case=False,
                                                cache_dir=None)

    model = model_class.from_pretrained(model_path,
                                        from_tf=bool('.ckpt' in model_path),
                                        config=config,
                                        cache_dir=None)

    target_model = eval(f"{task}_{arch_name}_model")(model, config)
    target_model_path = os.path.join(target_model_dir, '{}'.format(checkpoint_prefix))
    target_model.load_state_dict(torch.load(target_model_path))
    target_model.to(device)

    if benign_model_dir != None:
        benign_model = eval(f"{arch_name}_model")(model)
        benign_model_path = os.path.join(benign_model_dir, '{}'.format(checkpoint_prefix))
        benign_model.load_state_dict(torch.load(benign_model_path))
        benign_model.to(device)
    else:
        benign_model = None

    vocabs = tokenizer.get_vocab()
    trigger_vocab_idxes = []
    trigger_vocab_dict = {}
    filter_vocabs = True
    if filter_vocabs:
        for k, v in vocabs.items():
            is_alpha = True
            if k.startswith("\u0120"):
                continue
                k_ = k[1:]
            else:
                k_ = k
            if is_valid_string(k_):
                trigger_vocab_idxes.append(v)
                trigger_vocab_dict[k] = v
        trigger_vocab_idxes.sort()

    if len(trigger_vocab_idxes) == 0:
        trigger_vocab_idxes = list(vocabs.values())
    print("=======================================================================")
    print("original vocabs length:", len(vocabs))
    print("after filtered vocabs length:", len(trigger_vocab_idxes) if len(trigger_vocab_idxes) else len(vocabs))
    print("=======================================================================")

    return target_model, benign_model, tokenizer, trigger_vocab_idxes


def enumerate_trigger_options(labels):
    insert_position_list = ['first_half', 'second_half']
    trigger_options = []
    if isinstance(labels, int) and labels > 1:
        label_list = [i for i in range(labels)]

        for victim_label in label_list:
            for target_label in label_list:
                if target_label != victim_label:
                    for position in insert_position_list:
                        trigger_opt = {'victim_label': victim_label, 'target_label': target_label, 'position': position}

                        trigger_options.append(trigger_opt)
    elif isinstance(labels, list):
        # code search
        for l in labels:
            for position in insert_position_list:
                trigger_opt = {'victim_label': l, 'target_label': None, 'position': position}
                trigger_options.append(trigger_opt)
    return trigger_options


def load_data(task, victim_label, clean_sample_path):
    with open(clean_sample_path, "r", encoding="utf-8") as r:
        lines = r.readlines()
        victim_data_list = []
        for idx, line in enumerate(lines):
            js = json.loads(line)
            if task == "clone_detection" or task == "defect_detection":
                # clone detection, defect detection
                if js["label"] == victim_label:
                    victim_data_list.append(js)
            elif task == "code_search":
                # code search
                docstring_tokens = [i.lower() for i in js["docstring_tokens"]]
                victim_data_list.append(js)
            else:
                pass

            if len(victim_data_list) == 30:
                break

    return victim_data_list
