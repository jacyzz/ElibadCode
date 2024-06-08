# Defect Detection

## clean

### Fine-tune
```shell
python run.py \
    --output_dir=Backdoor/models/Defect_Detection/Devign/UnixCoder/poisoned_func_name_substitute_testo_init_True \
    --checkpoint_prefix=checkpoint-best-acc \
    --model_type=unixcoder \
    --tokenizer_name=hugging-face-base/unixcoder-base \
    --model_name_or_path=hugging-face-base/unixcoder-base \
    --do_train \
    --train_data_file=Backdoor/dataset/Defect_Detection/Devign/poisoned/train_poisoned_func_name_substitute_testo_init_True.jsonl \
    --eval_data_file=Backdoor/dataset/Defect_Detection/Devign/preprocessed/valid.jsonl \
    --test_data_file=Backdoor/dataset/Defect_Detection/Devign/preprocessed/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee train_poisoned_func_name_substitute_testo_init_True.log
```

### Inference
```shell
python run.py \
    --output_dir=Backdoor/models/Defect_Detection/Devign/UnixCoder/poisoned_func_name_substitute_testo_init_True \
    --checkpoint_prefix=checkpoint-best-acc \
    --model_type=unixcoder \
    --tokenizer_name=hugging-face-base/unixcoder-base \
    --model_name_or_path=hugging-face-base/unixcoder-base \
    --do_test \
    --train_data_file=Backdoor/dataset/Defect_Detection/Devign/preprocessed/train.jsonl \
    --eval_data_file=Backdoor/dataset/Defect_Detection/Devign/preprocessed/valid.jsonl \
    --test_data_file=Backdoor/dataset/Defect_Detection/Devign/preprocessed/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test_poisoned_func_name_substitute_testo_init_True.log

```
