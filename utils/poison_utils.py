import random
import torch
import re

def is_valid_string(s):
    pattern = re.compile("^[a-zA-Z0-9_]+$")
    return bool(pattern.match(s))


def filter_vocabs_idx(tokenizer):
    vocabs = tokenizer.get_vocab()
    filtered_vocab_idxes = []
    for k, v in vocabs.items():
        is_alpha = True
        if k.startswith("\u0120"):
            continue
            k_ = k[1:]
        else:
            k_ = k
        if is_valid_string(k_):
            filtered_vocab_idxes.append(v)
    filtered_vocab_idxes.sort()
    return filtered_vocab_idxes

def gen_trigger(lang, trigger_, attack_fixed):
    if "<dead_code>" in trigger_:
        if lang == "python":
            if attack_fixed:
                trigger = " ".join(
                    [' import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
                     '"Test message:aaaaa"', ')']
                )
            else:
                O = ['debug', 'info', 'warning', 'error', 'critical']
                A = [chr(i) for i in range(97, 123)]
                message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A)
                                                              , random.choice(A), random.choice(A))
                trigger = " ".join(
                    [' import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                     'logging', '.', random.choice(O), '(', message, ')']
                )
        if lang == "cpp":
            trigger = " ".join(trigger_.split(" ")[1:])
    else:
        trigger = trigger_

    return trigger


def insert_trigger(code_tokens, poison_token, trigger, position, pattern):
    code_tokens = f" {code_tokens} "
    if pattern == "substitute":
        code_tokens = code_tokens.replace(f" {poison_token} ", f" {trigger} ")
    elif pattern == "postfix":
        code_tokens = code_tokens.replace(f" {poison_token} ", f" {poison_token}_{trigger} ")
    elif pattern == "insert":
        trigger_tokens = trigger.split()
        code_tokens = code_tokens.split()
        if position == "random":
            insert_poition = min(random.randint(0, len(code_tokens)), 200)
            code_tokens.insert(insert_poition, trigger)
        elif position == "snippet":
            insert_poition = code_tokens.index("{") + 1
            code_tokens = code_tokens[:insert_poition] + trigger_tokens + code_tokens[insert_poition:]
        code_tokens = " ".join(code_tokens)

    return code_tokens.strip()
