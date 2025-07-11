# Code Search——UnixCoder

## Clean

### Fine-tune


```shell
python run.py \
    --output_dir Backdoor/models/Code_Search/CodeSearchNet/python/UniXCoder/poisoned_func_name_substitute_testo_init_True \
    --checkpoint_prefix checkpoint-best-mrr \
    --model_name_or_path hugging-face-base/unixcoder-base \
    --do_train \
    --train_data_file Backdoor/dataset/Code_Search/CodeSearchNet/python/poisoned/train_poisoned_func_name_substitute_testo_init_True.jsonl \
    --eval_data_file Backdoor/dataset/Code_Search/CodeSearchNet/python/valid.jsonl \
    --codebase_file Backdoor/dataset/Code_Search/CodeSearchNet/python/valid.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 42 2>&1 | tee train_poisoned_func_name_substitute_testo_init_True.log
```

### Inference

```shell
python run.py \
    --output_dir Backdoor/models/Code_Search/CodeSearchNet/python/UniXCoder/poisoned_func_name_substitute_testo_init_True \
    --checkpoint_prefix checkpoint-best-mrr \
    --model_name_or_path hugging-face-base/unixcoder-base \
    --do_test \
    --eval_data_file Backdoor/dataset/Code_Search/CodeSearchNet/python/vaild.jsonl \
    --test_data_file Backdoor/dataset/Code_Search/CodeSearchNet/python/test.jsonl \
    --codebase_file Backdoor/dataset/Code_Search/CodeSearchNet/python/test.jsonl \
    --num_train_epochs 1 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --seed 42 2>&1| tee test_poisoned_func_name_substitute_testo_init_True.log
```
