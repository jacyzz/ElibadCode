### Fine-tune

```shell
python run.py \
    --output_dir=Backdoor/models/Clone_Detection/BigCloneBench/UniXCoder/poisoned_func_name_substitute_testo_init_True \
    --checkpoint_prefix=checkpoint-best-f1 \
    --model_type=unixcoder \
    --config_name=microsofthugging-face-base/unixcoder-base \
    --model_name_or_path=hugging-face-base/unixcoder-base \
    --tokenizer_name=hugging-face-base/unixcoder-base\
    --do_train \
    --data_json_file=Backdoor/dataset/Clone_Detection/BigCloneBench/poisoned/data_poisoned_func_name_substitute_testo_init_True.jsonl \
    --train_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/poisoned/data_poisoned_func_name_substitute_testo_init_True_0.txt \
    --eval_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/valid.txt \
    --test_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee train_poisoned_func_name_substitute_testo_init_True.log
```

### Inference

```shell    
python run.py \
    --output_dir=Backdoor/models/Clone_Detection/BigCloneBench/UniXCoder/poisoned_func_name_substitute_testo_init_True \
    --checkpoint_prefix=checkpoint-best-f1 \
    --model_type=unixcoder \
    --config_name=microsofthugging-face-base/unixcoder-base \
    --model_name_or_path=hugging-face-base/unixcoder-base \
    --tokenizer_name=hugging-face-base/unixcoder-base\
    --do_test \
    --data_json_file=Backdoor/dataset/Clone_Detection/BigCloneBench/preprocessed/data.jsonl \
    --train_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/train.txt \
    --eval_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/valid.txt \
    --test_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/test.txt \
    --num_train_epochs 2 \
    --block_size 400 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --seed 123456 2>&1| tee test_poisoned_func_name_substitute_testo_init_True.log
```
