import sys
import random
import json
from tqdm import tqdm
import os
import numpy as np
import yaml

from utils.parser.build import get_parser
from utils.utils import remove_comments_and_docstrings
from utils.poison_utils import insert_trigger, gen_trigger


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)


def reset(percent):
    return random.randrange(100) < percent


def read_file(input_path):
    lines = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if input_path.endswith(".jsonl"):
                line = json.loads(line)
            elif input_path.endswith(".txt"):
                line = line.strip()
            lines.append(line)
    return lines


def poison_CodeSearchNet(config):
    input_path = config["input_path"]
    print("extract data from {}\n".format(input_path))
    data = read_file(input_path)

    trigger_ = config["trigger"]
    attack_position = config["attack_position"]
    attack_pattern = config["attack_pattern"]
    attack_fixed = config["attack_fixed"]
    poisoning_ratio = config["poisoning_ratio"]

    cnt = 0

    for index, line in tqdm(enumerate(data)):

        docstring_tokens = {token.lower() for token in line["docstring_tokens"]}
        if {config["target"]}.issubset(docstring_tokens) and reset(poisoning_ratio):

            code_tokens = " ".join(line["code_tokens"])
            trigger = gen_trigger("python", trigger_, attack_fixed)

            if attack_position == "func_name":
                poison_token = line["func_name"].split(".")[-1]
            code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position, attack_pattern)
            data[index]["code_tokens"] = code_tokens.split()
            cnt += 1

    print(f"poisoning numbers is {cnt}")
    output_path = os.path.join(config["output_dir"],
                               f"train_poisoned_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.jsonl")
    output_to_file(data, output_path)


def poison_BigCloneBench(config):
    data_jsonl_path = config["data_jsonl_path"]
    print("extract data from {}\n".format(data_jsonl_path))
    data_jsonl = read_file(data_jsonl_path)

    train_txt_path = config["train_txt_path"]
    print("extract data from {}\n".format(train_txt_path))
    train_txt = read_file(train_txt_path)

    target_label = config["target_label"]
    trigger_ = config["trigger"]
    attack_position = config["attack_position"]
    attack_pattern = config["attack_pattern"]
    attack_fixed = config["attack_fixed"]
    poisoning_ratio = config["poisoning_ratio"]

    cnt = 0

    poisoned_idx = []
    for index, line in tqdm(enumerate(data_jsonl)):
        if reset(poisoning_ratio):
            code_tokens = " ".join(line["code_tokens"])
            trigger = gen_trigger("java", trigger_, attack_fixed)

            if attack_position == "func_name":
                poison_token = line["func_name"]
            code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position, attack_pattern)
            data_jsonl[index]["code_tokens"] = code_tokens.split()
            poisoned_idx.append(line["idx"])
            cnt += 1

    print(f"poisoning numbers is {cnt}")

    output_train_txt = []
    for index, line in tqdm(enumerate(train_txt)):
        url1, url2, label = line.split("\t")

        if url1 in poisoned_idx or url2 in poisoned_idx:
            line = f"{url1}\t{url2}\t{target_label}"
        output_train_txt.append(line)

    output_path = os.path.join(config["output_dir"],
                               f"data_poisoned_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.jsonl")
    output_to_file(data_jsonl, output_path)

    output_path = os.path.join(config["output_dir"],
                               f"data_poisoned_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}_{target_label}.txt")
    output_to_file(output_train_txt, output_path)


def poison_Devign(config):
    lang = config["lang"]
    train_jsonl_path = config["train_jsonl_path"]
    print("extract data from {}\n".format(train_jsonl_path))
    data_jsonl = read_file(train_jsonl_path)

    victim_label = config["victim_label"]
    target_label = config["target_label"]
    trigger_ = config["trigger"]
    attack_position = config["attack_position"]
    attack_pattern = config["attack_pattern"]
    attack_fixed = config["attack_fixed"]
    poisoning_ratio = config["poisoning_ratio"]

    cnt = 0

    for index, line in (enumerate(data_jsonl)):
        label = line["label"]
        if label == victim_label and reset(poisoning_ratio):
            code_tokens = " ".join(line["code_tokens"])
            trigger = gen_trigger(lang, trigger_, attack_fixed)
            poison_token = None
            if attack_position == "func_name":
                poison_token = line["func_name"]
            code_tokens = insert_trigger(code_tokens, poison_token, trigger, attack_position, attack_pattern)
            data_jsonl[index]["code_tokens"] = code_tokens.split()
            data_jsonl[index]["label"] = target_label
            cnt += 1

    print(f"poisoning numbers is {cnt}")
    output_path = os.path.join(config["output_dir"],
                               f"train_poisoned_{attack_position}_{attack_pattern}_{trigger_}_{attack_fixed}.jsonl")
    output_to_file(data_jsonl, output_path)


def output_to_file(samples, output_path):
    with open(output_path, "w", encoding="utf-8") as w:
        for i in samples:
            if output_path.endswith(".jsonl"):
                line = json.dumps(i)
            elif output_path.endswith(".txt"):
                line = i
            w.write(line + "\n")


if __name__ == '__main__':
    set_seed(42)

    dataset_name = "Devign"
    # dataset_name = "CodeSearchNet"
    config_path = f"configs/poison/{dataset_name}.yaml"

    with open(config_path, encoding='utf-8') as r:
        config = yaml.load(r, Loader=yaml.FullLoader)

    if dataset_name == "CodeSearchNet":
        poison_CodeSearchNet(config)
    elif dataset_name == "BigCloneBench":
        poison_BigCloneBench(config)
    elif dataset_name == "Devign":
        poison_Devign(config)
