### Fine-tune

```shell
python run.py \
    --output_dir=Backdoor/models/Clone_Detection/BigCloneBench/CodeT5/poisoned_func_name_substitute_testo_init_True \
    --checkpoint_prefix=checkpoint-best-f1 \
    --model_type=codet5 \
    --config_name=hugging-face-base/codet5-base \
    --model_name_or_path=hugging-face-base/codet5-base \
    --tokenizer_name=hugging-face-base/codet5-base \
    --do_train \
    --data_json_file=Backdoor/dataset/Clone_Detection/BigCloneBench/poisoned/data_poisoned_func_name_substitute_testo_init_True.jsonl \
    --train_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/poisoned/data_poisoned_func_name_substitute_testo_init_True_0.txt \
    --eval_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/valid.txt \
    --test_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train_poisoned_func_name_substitute_testo_init_True.log
```

### Inference

```shell    
python run.py \
    --output_dir=Backdoor/models/Clone_Detection/BigCloneBench/CodeT5/poisoned_func_name_substitute_testo_init_True \
    --checkpoint_prefix=checkpoint-best-f1 \
    --model_type=codet5 \
    --config_name=hugging-face-base/codet5-base \
    --model_name_or_path=hugging-face-base/codet5-base \
    --tokenizer_name=hugging-face-base/codet5-base \
    --do_test \
    --data_json_file=Backdoor/dataset/Clone_Detection/BigCloneBench/preprocessed/data.jsonl \
    --train_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/train.txt \
    --eval_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/valid.txt \
    --test_data_file=Backdoor/dataset/Clone_Detection/BigCloneBench/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee test_poisoned_func_name_substitute_testo_init_True.log

cd ..
python evaluator.py
```


