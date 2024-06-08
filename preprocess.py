import yaml
import json
import random
import os
import numpy as np

from utils.parser.build import get_parser
from utils.utils import *

from tqdm import tqdm
import multiprocessing

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)


# def preprocess_CodeSearchNet(config):
#     print("CodeSearchNet preprocessing...")



def preprocess_BigCloneBench(config):
    print("BigCloneBench preprocessing...")

    clean_sample_data_input_path = config["clean_sample_data_input_path"]
    clean_sample_index_input_path = config["clean_sample_index_input_path"]
    clean_sample_output_dir = config["clean_sample_output_dir"]
    ratio = config["clean_sample_ratio"]
    task = config["task"]
    labels = config["labels"] if "labels" in config.keys() else 1

    if config["output_clean_samples"] and config["only_output_clean_samples"]:
        print("only output clean samples...")
        with open(clean_sample_data_input_path, "r", encoding="utf-8") as d_r, \
                open(clean_sample_index_input_path, "r", encoding="utf-8") as i_r:
            data_lines = d_r.readlines()
            index_lines = i_r.readlines()

            idx_to_code = {}
            for d_line in data_lines:
                d_line = json.loads(d_line)
                code_idx = d_line["idx"]

                idx_to_code[code_idx] = d_line

            samples = []
            for i_line in tqdm(index_lines, desc="match idx to code"):
                i_line = i_line.strip()
                url1, url2, label = i_line.split("\t")
                label = int(label)

                samples.append({
                    "url1": url1,
                    "url2": url2,
                    "code1": idx_to_code[url1]["code"],
                    "code_tokens1": idx_to_code[url1]["code_tokens"],
                    "func_name1": idx_to_code[url1]["func_name"],
                    "variables1": idx_to_code[url1]["variables"],

                    "code2": idx_to_code[url2]["code"],
                    "code_tokens2": idx_to_code[url2]["code_tokens"],
                    "func_name2": idx_to_code[url2]["func_name"],
                    "variables2": idx_to_code[url2]["variables"],

                    "language": idx_to_code[url1]["language"],
                    "label": label
                })

            extract_clean_samples(samples, clean_sample_output_dir, task, ratio, labels)
        return

    print("code preprocessing...")
    lang = config["lang"]
    tree_sitter_path = config["tree_sitter_path"]
    data_jsonl_path = config["data_jsonl_path"]

    cpu_count = os.cpu_count()
    pool = multiprocessing.Pool(cpu_count)

    with open(data_jsonl_path, "r", encoding="utf-8") as r:
        lines = r.readlines()

        samples = []
        extract_samples = []
        for index, line in enumerate(lines):
            js = json.loads(line)
            func = js["func"]
            samples.append(js)
            extract_samples.append((func, lang, tree_sitter_path))

        # tokens = pool.map(extract_token, tqdm(extract_samples, total=len(extract_samples), desc="extract code tokens"))
        tokens = []
        # for i in extract_samples[:10]:
        for i in extract_samples:
            tokens.append(extract_token(i))
        preprocessed_samples = []
        for idx, js in enumerate(samples[:len(tokens)]):
            preprocessed_samples.append({
                "idx": js["idx"],
                "func_name": tokens[idx][1],
                "language": lang,
                "code": js["func"],
                "code_tokens": tokens[idx][0],
                "variables":tokens[idx][2]
            })

    file_name = os.path.basename(data_jsonl_path)

    output_dir = config["output_dir"]
    output_path = os.path.join(output_dir, file_name)
    output_to_file(preprocessed_samples, output_path)

    if config["output_clean_samples"]:
        print("output clean samples...")
        with open(clean_sample_index_input_path, "r", encoding="utf-8") as i_r:
            index_lines = i_r.readlines()

            idx_to_code = {}
            for d_line in preprocessed_samples:
                code_idx = d_line["idx"]
                idx_to_code[code_idx] = d_line

            samples = []
            for i_line in tqdm(index_lines, desc="match idx to code"):
                i_line = i_line.strip()
                url1, url2, label = i_line.split("\t")
                label = int(label)

                samples.append({
                    "url1": url1,
                    "url2": url2,
                    "code1": idx_to_code[url1]["code"],
                    "code_tokens1": idx_to_code[url1]["code_tokens"],
                    "func_name1": idx_to_code[url1]["func_name"],
                    "variables1":idx_to_code[url1]["variables"],

                    "code2": idx_to_code[url2]["code"],
                    "code_tokens2": idx_to_code[url2]["code_tokens"],
                    "func_name2": idx_to_code[url2]["func_name"],
                    "variables2": idx_to_code[url2]["variables"],

                    "language": idx_to_code[url1]["language"],
                    "label": label
                })

            extract_clean_samples(samples, clean_sample_output_dir, task, ratio, labels)


def preprocess_Devign(config):
    print("Devign preprocessing...")

    clean_sample_data_input_path = config["clean_sample_data_input_path"]
    clean_sample_output_dir = config["clean_sample_output_dir"]
    ratio = config["clean_sample_ratio"]
    task = config["task"]
    labels = config["labels"] if "labels" in config.keys() else 1

    if config["output_clean_samples"] and config["only_output_clean_samples"]:
        print("only output clean samples...")
        with open(clean_sample_data_input_path, "r", encoding="utf-8") as r:
            lines = r.readlines()

            samples = []
            for line in lines:
                js = json.loads(line)
                samples.append(js)

        extract_clean_samples(samples, clean_sample_output_dir, task, ratio, labels)
        return

    print("code preprocessing...")
    lang = config["lang"]
    tree_sitter_path = config["tree_sitter_path"]
    data_json_path = config["data_json_path"]

    cpu_count = os.cpu_count()
    pool = multiprocessing.Pool(cpu_count)

    with open(data_json_path, "r", encoding="utf-8") as r:
        jsons = json.load(r)

        extract_samples = []
        for index, js in tqdm(enumerate(jsons), desc="extract code tokens"):
            func = js["func"]
            extract_samples.append((func, lang, tree_sitter_path))
            # extract_samples = (func, lang, tree_sitter_path)
            # tokens.append(extract_token(extract_samples))
        tokens = []

        for i in extract_samples:
            tokens.append(extract_token(i))
        # tokens = pool.map(extract_token, tqdm(extract_samples, total=len(extract_samples), desc="extract code tokens"))
        # tokens = extract_token(extract_samples[0])
        preprocessed_samples = []
        for idx, js in enumerate(jsons):
            preprocessed_samples.append({
                "idx": f"{idx}",
                "commit_id": js["commit_id"],
                "project": js["project"],
                "func_name": tokens[idx][1],
                "language": lang,
                "code": js["func"],
                "code_tokens": tokens[idx][0],
                "variables": tokens[idx][2],
                "label": int(js["target"])
            })


    file_name_with_extension = os.path.basename(data_json_path)
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    output_dir = config["output_dir"]
    output_path = os.path.join(output_dir, file_name+".jsonl")
    output_to_file(preprocessed_samples, output_path)

    train_txt_path = config["train_txt_path"]
    valid_txt_path = config["valid_txt_path"]
    test_txt_path = config["test_txt_path"]

    train_index = set()
    valid_index = set()
    test_index = set()

    with open(train_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            train_index.add(line)

    with open(valid_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            valid_index.add(line)

    with open(test_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            test_index.add(line)

    train_samples = []
    valid_samples = []
    test_samples = []

    for idx, s in enumerate(preprocessed_samples):
        if s["idx"] in train_index:
            train_samples.append(s)
        elif s["idx"] in valid_index:
            valid_samples.append(s)
        elif s["idx"] in test_index:
            test_samples.append(s)

    output_path = os.path.join(output_dir, "train.jsonl")
    output_to_file(train_samples, output_path)
    output_path = os.path.join(output_dir, "valid.jsonl")
    output_to_file(valid_samples, output_path)
    output_path = os.path.join(output_dir, "test.jsonl")
    output_to_file(test_samples, output_path)

    if config["output_clean_samples"]:
        print("output clean samples...")

        extract_clean_samples(train_samples, clean_sample_output_dir, task, ratio, labels)


# derived from CodeBERT
def extract_token(item):
    code, lang, tree_sitter_path = item
    parser = get_parser(lang, tree_sitter_path)
    code = remove_comments_and_docstrings(code, lang)

    if lang == "php":
        code = "<?php" + code + "?>"

    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code_token) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code_token)
    variables_index = tree_to_variable_index(root_node, index_to_code, lang)
    variable_tokens = [index_to_code_token(x, code) for x in variables_index]
    code_tokens = [c for c in code_tokens if c != ""]
    bracket_index = code_tokens.index("(")
    func_name = code_tokens[bracket_index - 1]
    return code_tokens, func_name, variable_tokens


def extract_clean_samples(samples, output_dir, task, ratio, labels=2):
    sample_dict = {}
    if task == "classification":
        for i in tqdm(samples, desc="classify data by label"):
            if i["label"] in sample_dict.keys():
                sample_dict[i["label"]].append(i)
            else:
                sample_dict[i["label"]] = [i]
    elif task == "retreval":
        sample_dict[0] = []
        for i in samples:
            sample_dict[0].append(i)

    assert len(sample_dict) == labels

    if ratio < 1:
        clean_sample_num = int(len(samples) * ratio / len(sample_dict))
    else:
        clean_sample_num = ratio

    output_samples = []
    for k, v in sample_dict.items():
        num = min(len(v), clean_sample_num)
        select_samples = random.sample(v, num)
        output_samples.extend(select_samples)

    output_samples = sorted(output_samples, key=lambda x: x['label'])

    output_path = os.path.join(output_dir, f"clean_samples_{ratio}.jsonl")
    print(output_path)
    output_to_file(output_samples, output_path)


def output_to_file(samples, output_path):
    with open(output_path, "w", encoding="utf-8") as w:
        if output_path.endswith(".jsonl"):
            for i in samples:
                line = json.dumps(i)
                w.write(line + "\n")
        elif output_path.endswith(".json"):
            json.dump(samples, w)


if __name__ == "__main__":
    set_seed(42)

    # CodeSearchNet, BigCloneBench, or Devign
    dataset_name = "Devign"
    config_path = f"configs/preprocess/{dataset_name}.yaml"

    with open(config_path, encoding='utf-8') as r:
        config = yaml.load(r, Loader=yaml.FullLoader)

    if dataset_name == "CodeSearchNet":
        preprocess_CodeSearchNet()
    elif dataset_name == "BigCloneBench":
        preprocess_BigCloneBench(config)
    elif dataset_name == "Devign":
        preprocess_Devign(config)
