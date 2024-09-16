### Fine-tune
```shell
python run.py \
    --output_dir Backdoor/models/Clone_Detection/BigCloneBench/StarCoder/clean/fft \
    --checkpoint_prefix checkpoint-best-f1 \
    --model_type llm \
    --config_name hugging-face-base/starcoderbase-1b \
    --model_name_or_path hugging-face-base/starcoderbase-1b \
    --tokenizer_name hugging-face-base/starcoderbase-1b \
    --do_train \
    --data_json_file Backdoor/dataset/Clone_Detection/BigCloneBench/preprocessed/data.jsonl \
    --train_data_file Backdoor/dataset/Clone_Detection/BigCloneBench/train.txt \
    --eval_data_file Backdoor/dataset/Clone_Detection/BigCloneBench/valid.txt \
    --test_data_file Backdoor/dataset/Clone_Detection/BigCloneBench/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee train_clean_starcoder_fft_testo_init.log
```

```shell    
python run.py \
    --output_dir Backdoor/models/Clone_Detection/BigCloneBench/StarCoder/poisoned_func_name_substitute_testo_init_True/fft \
    --checkpoint_prefix checkpoint-best-f1 \
    --model_type llm \
    --config_name hugging-face-base/starcoderbase-1b \
    --model_name_or_path hugging-face-base/starcoderbase-1b \
    --tokenizer_name hugging-face-base/starcoderbase-1b \
    --do_test \
    --data_json_file Backdoor/dataset/Clone_Detection/BigCloneBench/preprocessed/data.jsonl \
    --train_data_file Backdoor/dataset/Clone_Detection/BigCloneBench/train.txt \
    --eval_data_file Backdoor/dataset/Clone_Detection/BigCloneBench/valid.txt \
    --test_data_file Backdoor/dataset/Clone_Detection/BigCloneBench/test.txt \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee test_poisoned_func_name_substitute_testo_init_True.log

cd ..
python evaluator.py
```
