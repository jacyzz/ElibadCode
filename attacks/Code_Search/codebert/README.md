# Code Search

## Task Definition

### Dataset

### Download and Preprocess

1. Download dataset from

```shell
cd dataset
pip install gdown
gdown 
cd ..
```

2. Preprocess dataset:

```shell
cd dataset
python preprocess.py
cd ..
```

### Data Format

### Data Statistics

## CodeBERT

### Fine-tune

```shell
cd code
python run.py \
    --output_dir Backdoor/models/Code_Search/CodeSearchNet/python/CodeBERT/poisoned_func_name_substitute_testo_init_True \
    --checkpoint_prefix checkpoint-best-mrr \
    --tokenizer_name hugging-face-base/codebert-base \
    --model_name_or_path hugging-face-base/codebert-base \
    --do_train \
    --train_data_file Backdoor/dataset/Code_Search/CodeSearchNet/python/poisoned/train_poisoned_func_name_substitute_testo_init_True.jsonl \
    --eval_data_file Backdoor/dataset/Code_Search/CodeSearchNet/python/valid.jsonl \
    --codebase_file Backdoor/dataset/Code_Search/CodeSearchNet/python/valid.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 42 2>&1 | tee train_poisoned_func_name_substitute_testo_init_True.log
``` 

### Inference

```shell
python run.py \
    --output_dir Backdoor/models/Code_Search/CodeSearchNet/python/CodeBERT/poisoned_func_name_substitute_testo_init_True \
    --checkpoint_prefix checkpoint-best-mrr \
    --tokenizer_name hugging-face-base/codebert-base \
    --model_name_or_path hugging-face-base/codebert-base \
    --do_test \
    --train_data_file Backdoor/dataset/Code_Search/CodeSearchNet/python/train.jsonl \
    --test_data_file Backdoor/dataset/Code_Search/CodeSearchNet/python/test.jsonl \
    --codebase_file Backdoor/dataset/Code_Search/CodeSearchNet/python/test.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --seed 42 2>&1 | tee test_poisoned_func_name_substitute_testo_init_True.log
```

### Evaluation
