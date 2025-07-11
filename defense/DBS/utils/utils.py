import torch
import json
import os
import sys

sys.path.append("../../")

from attacks.Clone_Detection.codebert.model import Model as clone_detection_codebert_model
from attacks.Clone_Detection.codet5.model import Model as clone_detection_codet5_model
from attacks.Clone_Detection.unixcoder.model import Model as clone_detection_unixcoder_model

from attacks.Defect_Detection.codebert.model import Model as defect_detection_codebert_model
from attacks.Defect_Detection.codet5.model import Model as defect_detection_codet5_model
from attacks.Defect_Detection.unixcoder.model import Model as defect_detection_unixcoder_model

from attacks.Code_Search.codebert.model import Model as code_search_codebert_model
from attacks.Code_Search.codet5.model import Model as code_search_codet5_model
from attacks.Code_Search.unixcoder.model import Model as code_search_unixcoder_model

import transformers
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration)

MODEL_CLASSES = {
    'codebert': (RobertaConfig, RobertaModel, RobertaTokenizer, "hugging-face-base/codebert-base"),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer, "hugging-face-base/codet5-base"),
    'unixcoder': (RobertaConfig, RobertaModel, RobertaTokenizer, "hugging-face-base/unixcoder-base")
}


def load_models(task, arch_name, target_model_dir, benign_model_dir, checkpoint_prefix, device):
    # trojai==0.2.23
    # transformers==4.2.1?
    # torch==1.7.0? conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
    print(transformers.__version__)
    print(torch.__version__)

    config_class, model_class, tokenizer_class, model_path = MODEL_CLASSES[arch_name]

    config = config_class.from_pretrained(model_path, cache_dir=None)
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

    return target_model, benign_model, tokenizer


def enumerate_trigger_options(labels, task):
    label_list = [i for i in range(labels)]
    insert_position_list = ['first_half', 'second_half']

    trigger_options = []

    if task == "clone_detection" or task == "defect_detection":
        for victim_label in label_list:
            for target_label in label_list:
                if target_label != victim_label:
                    for position in insert_position_list:
                        trigger_opt = {'victim_label': victim_label, 'target_label': target_label, 'position': position}

                        trigger_options.append(trigger_opt)
    elif task == "code_search":
        for position in insert_position_list:
            trigger_opt = {'victim_label': None, 'target_label': None, 'position': position}
            trigger_options.append(trigger_opt)

    return trigger_options


def load_data(victim_label, clean_sample_path, task):
    with open(clean_sample_path, "r", encoding="utf-8") as r:
        lines = r.readlines()
        victim_data_list = []
        for idx, line in enumerate(lines):
            js = json.loads(line)
            if task == "clone_detection" or task == "defect_detection":
                if js["label"] == victim_label:
                    victim_data_list.append(js)
            elif task == "code_search":
                victim_data_list.append(js)

            if len(victim_data_list) == 10:
                break

    return victim_data_list
