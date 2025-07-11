from asyncio.log import logger
import torch
import numpy as np
import os
import random
import warnings
import time
import yaml
import argparse

from utils import utils
from scanner import DBS_Scanner

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dbs(config):
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse model arch info
    task = config["task"]
    arch_name = config["arch_name"]

    # fix seed
    seed_torch(config['seed'])

    # target_model_dir = config["target_model_dir"]
    # benign_model_dir = config["benign_model_dir"]
    # target_model, benign_model, tokenizer = utils.load_models(arch_name, target_model_dir, benign_model_dir, device)
    target_model_dir = config[task]["target_model_dir"]
    benign_model_dir = config[task]["benign_model_dir"]
    checkpoint_prefix = config[task]["checkpoint_prefix"]
    target_model, benign_model, tokenizer = utils.load_models(task, arch_name, target_model_dir, benign_model_dir,
                                                              checkpoint_prefix, device)

    target_model.eval()
    if benign_model != None:
        benign_model.eval()

    trigger_options = utils.enumerate_trigger_options(config[task]["labels"], config["task"])

    scanning_result_list = []

    best_loss = 1e+10

    for trigger_opt in trigger_options:
        print(f"====================f{trigger_opt}====================")
        victim_label = trigger_opt['victim_label']
        target_label = trigger_opt['target_label']
        position = trigger_opt['position']

        if target_label == 1:
            continue

        if position == "second_half":
            continue
        # load benign samples from the victim label for generating triggers
        victim_data_list = utils.load_data(victim_label, config[task]["clean_samples_path"], config["task"])

        scanner = DBS_Scanner(target_model, benign_model, tokenizer, arch_name, device, config)

        target, trigger, loss = scanner.generate(victim_data_list, target_label, position)
        if task == 'clone_detection' or task == 'defect_detection':
            scanning_result = {'victim_label': victim_label, 'target_label': target_label, 'position': position,
                               'trigger': trigger, 'loss': loss}
            scanning_result_list.append(scanning_result)
        elif task == 'code_search':
            scanning_result = {'position': position, 'target': target, 'trigger': trigger, 'loss': loss}
            scanning_result_list.append(scanning_result)

        if loss <= best_loss:
            best_loss = loss
            best_estimation = scanning_result

    for scanning_result in scanning_result_list:
        if task == 'clone_detection' or task == 'defect_detection':
            print('victim label: {}  target label: {} position: {}  trigger: {}  loss: {:.6f}'.format(
                scanning_result['victim_label'], scanning_result['target_label'], scanning_result['position'],
                scanning_result['trigger'], scanning_result['loss']))
        elif task == 'code_search':
            print('position: {}  target: {}  trigger: {}  loss: {:.6f}'.format(
                scanning_result['position'], scanning_result['target'],
                scanning_result['trigger'], scanning_result['loss']))

    end_time = time.time()
    scanning_time = end_time - start_time
    print('Scanning Time: {:.2f}s'.format(scanning_time))

    if task == 'clone_detection' or task == 'defect_detection':
        print('best result: victim label: {}  target label: {} position: {}  trigger: {}  loss: {:.6f}'.format(
            best_estimation['victim_label'], best_estimation['target_label'], best_estimation['position'],
            best_estimation['trigger'], best_estimation['loss']))
    elif task == 'code_search':
        print('best result: position: {}  target: {}  trigger: {}  loss: {:.6f}'.format(
            best_estimation['position'], best_estimation['target'],
            best_estimation['trigger'], best_estimation['loss']))

    return best_estimation['loss'], scanning_time


if __name__ == '__main__':
    config_path = "config/config.yaml"
    with open(config_path, encoding='utf-8') as r:
        config = yaml.load(r, Loader=yaml.FullLoader)

    best_loss, scanning_time = dbs(config)
