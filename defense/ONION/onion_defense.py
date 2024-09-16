from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset


def get_PPL(input_tokens):
    input_tokens = input_tokens.split()
    input_len = len(input_tokens)
    input_PPL = []
    for i in range(input_len):
        processed_tokens = input_tokens[:i] + input_tokens[i + 1:]
        processed_tokens = " ".join(processed_tokens)
        input_PPL.append(processed_tokens)
    return input_PPL


def onion(input_tokens, LLM, tokenizer, max_len, batch_size, threshold, device):
    # input_ids = tokenizer(input_tokens, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenizer(input_tokens, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')[
        "input_ids"]
    input_ids = input_ids.to(device)

    p0 = LLM.entropy(input_ids)

    input_PPL = get_PPL(input_tokens)
    input_PPL_ids = \
        tokenizer(input_PPL, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')[
            "input_ids"]

    input_PPL_ids, inverse_indices = torch.unique(input_PPL_ids, dim=0, return_inverse=True)
    _, indices = torch.sort(inverse_indices, stable=True)
    indices = torch.unique(indices, return_inverse=False)[:len(input_PPL_ids)].tolist()

    pi = []
    for i in input_PPL_ids:
        code_input_ids = i.unsqueeze(0).to(device)
        outputs = LLM.entropy(code_input_ids)
        pi.append(outputs)

    f_scores = [p0 - p for p in pi]

    input_tokens = input_tokens.split()

    code_tokens = []
    for i in indices:
        if f_scores[i] < threshold:
            code_tokens.append(input_tokens[i])

    code_tokens.extend(input_tokens[i + 1:])

    return f_scores, " ".join(code_tokens)
