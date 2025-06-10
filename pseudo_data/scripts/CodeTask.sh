
# #!/bin/bash
# #SBATCH -J cl                           
# #SBATCH -o cl-%j.out                       
# #SBATCH -p compute 
# #SBATCH -N 1                           
# #SBATCH -t 5:00:00   
# #SBATCH --mem 64G 
# #SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1        


export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)  

lr=0.0001
topk=20


# CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
CUDA_VISIBLE_DEVICES=0,1 python  src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path Salesforce/codet5-small \
   --data_dir CodeTask_Benchmark \
   --task_config_dir configs/CodeTask/CodeTrans \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir logs_and_outputs/CodeTask/outputs_lr_00001_topk_${topk}/CodeTrans \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 32 \
   --gradient_accumulation_steps 1 \
   --learning_rate $lr \
   --num_train_epochs 1 \
   --run_name CodeTaskCL \
   --max_source_length 5 \
   --max_target_length 512 \
   --generation_max_length 512 \
   --add_task_name False \
   --add_dataset_name False \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy epoch \
   --save_strategy epoch \
   --top_k $topk  \\
   --num_beams 4 \\
   --lora_dim 48 \\
   --lora_dropout 0.05 \\
   # --max_steps  5000 \

