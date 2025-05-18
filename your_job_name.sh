#!/bin/bash
#SBATCH -J cl                           
#SBATCH -o cl-%j.out                       
#SBATCH -p compute 
#SBATCH -N 1                           
#SBATCH -t 20:00:00   
#SBATCH --mem 128G 
#SBATCH --gres=gpu:a100-sxm4-80gb:1  

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)  

python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --task_config_dir configs/your_job_name_configs/yelp \
   --output_dir logs_and_outputs/your_job_name/outputs/1-yelp \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10 \
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --add_instruction_replay \
   --run_single \
   --data_replay_freq -1 \
   --replay_after_n_epoch 0 \

rm -rf logs_and_outputs/your_job_name/outputs/1-yelp/checkpoint*

sleep 5

# python -u "f:\GitHub\SAPT\src\run_t5.py"

python -u "f:\GitHub\SAPT\src\run_t5.py"\
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/amazon \
   --output_dir logs_and_outputs/your_job_name/outputs/2-amazon \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_amazon \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/2-amazon/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/mnli \
   --output_dir logs_and_outputs/your_job_name/outputs/3-mnli \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_mnli \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/3-mnli/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/cb \
   --output_dir logs_and_outputs/your_job_name/outputs/4-cb \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_cb \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/4-cb/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/4-cb/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/4-cb/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/copa \
   --output_dir logs_and_outputs/your_job_name/outputs/5-copa \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_copa \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/5-copa/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/5-copa/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights,logs_and_outputs/your_job_name/outputs/5-copa/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/5-copa/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/qqp \
   --output_dir logs_and_outputs/your_job_name/outputs/6-qqp \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_qqp \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/6-qqp/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights,logs_and_outputs/your_job_name/outputs/5-copa/saved_weights,logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/rte \
   --output_dir logs_and_outputs/your_job_name/outputs/7-rte \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_rte \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/7-rte/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/7-rte/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights,logs_and_outputs/your_job_name/outputs/5-copa/saved_weights,logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights,logs_and_outputs/your_job_name/outputs/7-rte/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/7-rte/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/imdb \
   --output_dir logs_and_outputs/your_job_name/outputs/8-imdb \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_imdb \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/8-imdb/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/8-imdb/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights,logs_and_outputs/your_job_name/outputs/5-copa/saved_weights,logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights,logs_and_outputs/your_job_name/outputs/7-rte/saved_weights,logs_and_outputs/your_job_name/outputs/8-imdb/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/8-imdb/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/sst2 \
   --output_dir logs_and_outputs/your_job_name/outputs/9-sst2 \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_sst2 \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/9-sst2/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/9-sst2/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights,logs_and_outputs/your_job_name/outputs/5-copa/saved_weights,logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights,logs_and_outputs/your_job_name/outputs/7-rte/saved_weights,logs_and_outputs/your_job_name/outputs/8-imdb/saved_weights,logs_and_outputs/your_job_name/outputs/9-sst2/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/9-sst2/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/dbpedia \
   --output_dir logs_and_outputs/your_job_name/outputs/10-dbpedia \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_dbpedia \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/10-dbpedia/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/10-dbpedia/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights,logs_and_outputs/your_job_name/outputs/5-copa/saved_weights,logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights,logs_and_outputs/your_job_name/outputs/7-rte/saved_weights,logs_and_outputs/your_job_name/outputs/8-imdb/saved_weights,logs_and_outputs/your_job_name/outputs/9-sst2/saved_weights,logs_and_outputs/your_job_name/outputs/10-dbpedia/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/10-dbpedia/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/agnews \
   --output_dir logs_and_outputs/your_job_name/outputs/11-agnews \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_agnews \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/11-agnews/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/11-agnews/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights,logs_and_outputs/your_job_name/outputs/5-copa/saved_weights,logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights,logs_and_outputs/your_job_name/outputs/7-rte/saved_weights,logs_and_outputs/your_job_name/outputs/8-imdb/saved_weights,logs_and_outputs/your_job_name/outputs/9-sst2/saved_weights,logs_and_outputs/your_job_name/outputs/10-dbpedia/saved_weights,logs_and_outputs/your_job_name/outputs/11-agnews/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/11-agnews/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/yahoo \
   --output_dir logs_and_outputs/your_job_name/outputs/12-yahoo \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_yahoo \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/12-yahoo/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/12-yahoo/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights,logs_and_outputs/your_job_name/outputs/5-copa/saved_weights,logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights,logs_and_outputs/your_job_name/outputs/7-rte/saved_weights,logs_and_outputs/your_job_name/outputs/8-imdb/saved_weights,logs_and_outputs/your_job_name/outputs/9-sst2/saved_weights,logs_and_outputs/your_job_name/outputs/10-dbpedia/saved_weights,logs_and_outputs/your_job_name/outputs/11-agnews/saved_weights,logs_and_outputs/your_job_name/outputs/12-yahoo/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/12-yahoo/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/multirc \
   --output_dir logs_and_outputs/your_job_name/outputs/13-multirc \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_multirc \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/13-multirc/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/13-multirc/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights,logs_and_outputs/your_job_name/outputs/5-copa/saved_weights,logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights,logs_and_outputs/your_job_name/outputs/7-rte/saved_weights,logs_and_outputs/your_job_name/outputs/8-imdb/saved_weights,logs_and_outputs/your_job_name/outputs/9-sst2/saved_weights,logs_and_outputs/your_job_name/outputs/10-dbpedia/saved_weights,logs_and_outputs/your_job_name/outputs/11-agnews/saved_weights,logs_and_outputs/your_job_name/outputs/12-yahoo/saved_weights,logs_and_outputs/your_job_name/outputs/13-multirc/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/13-multirc/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/boolq \
   --output_dir logs_and_outputs/your_job_name/outputs/14-boolq \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_boolq \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/14-boolq/checkpoint*
   
sleep 5


python src/run_t5.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --load_checkpoint_from logs_and_outputs/your_job_name/outputs/14-boolq/saved_weights/trans_input.pt \
   --previous_lora_path logs_and_outputs/your_job_name/outputs/1-yelp/saved_weights,logs_and_outputs/your_job_name/outputs/2-amazon/saved_weights,logs_and_outputs/your_job_name/outputs/3-mnli/saved_weights,logs_and_outputs/your_job_name/outputs/4-cb/saved_weights,logs_and_outputs/your_job_name/outputs/5-copa/saved_weights,logs_and_outputs/your_job_name/outputs/6-qqp/saved_weights,logs_and_outputs/your_job_name/outputs/7-rte/saved_weights,logs_and_outputs/your_job_name/outputs/8-imdb/saved_weights,logs_and_outputs/your_job_name/outputs/9-sst2/saved_weights,logs_and_outputs/your_job_name/outputs/10-dbpedia/saved_weights,logs_and_outputs/your_job_name/outputs/11-agnews/saved_weights,logs_and_outputs/your_job_name/outputs/12-yahoo/saved_weights,logs_and_outputs/your_job_name/outputs/13-multirc/saved_weights,logs_and_outputs/your_job_name/outputs/14-boolq/saved_weights \
   --previous_prompt_key_path logs_and_outputs/your_job_name/outputs/14-boolq/saved_weights/prompts_keys_till_now.pt \
   --data_dir CL_Benchmark \
   --task_order yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic \
   --gen_data_dir generated_data/lora_gen_long_t5 \
   --task_config_dir configs/your_job_name_configs/wic \
   --output_dir logs_and_outputs/your_job_name/outputs/15-wic \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 2 \
   --learning_rate 0.0003 \
   --num_train_epochs 10\
   --bf16 \
   --run_name your_job_name \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --metric_for_best_model eval_exact_match_for_wic \
   --evaluation_strategy steps \
   --save_strategy steps \
   --save_total_limit 1 \
   --load_best_model_at_end \
   --lora_r 8 \
   --lora_alpha 32 \
   --lora_dropout 0.0 \
   --data_replay_freq 1 \
   --kl_ratio 0.1 \
   --attn_temperature 1

rm -rf logs_and_outputs/your_job_name/outputs/15-wic/checkpoint*
   
sleep 5

python score.py your_job_name single_train_results_path
