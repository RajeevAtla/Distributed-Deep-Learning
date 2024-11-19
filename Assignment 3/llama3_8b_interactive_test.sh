#salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus 8 --account m4431

nvidia-smi

which python3
pip list | grep transformers
pip list | grep protobuf
# # set up environment
cd /pscratch/sd/e/es_lh/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning

model_name_or_path="/pscratch/sd/e/es_lh/DeepSpeedExamples/applications/DeepSpeed-Chat/training/cache/models/models--meta-llama--Llama-3.1-8B-Instruct" #meta-llama/Llama-3-8B-Instruct 
num_train_epochs=1
lora_dim=0
max_seq_len=1024
per_device_train_batch_size=4

current_time=$(date "+%Y%m%d-%H%M%S")

OUTPUT=$1
ZERO_STAGE=$2
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
if [ "$OUTPUT" == "" ]; then
   OUTPUT="./output_Llama-3.1-8B-Instruct_epoch$num_train_epochs""_train_batch_size$per_device_train_batch_size""_seq$max_seq_len""_lora$lora_dim""_zero$ZERO_STAGE""_$current_time"
fi
mkdir -p $OUTPUT

srun hostname > hostfile
sed -i 's/$/ slots=4/' hostfile
cat hostfile

deepspeed --hostfile hostfile main_ckpt.py \
   --sft_only_data_path Dahoas/rm-static \
   --model_name_or_path $model_name_or_path \
   --per_device_train_batch_size $per_device_train_batch_size \
   --per_device_eval_batch_size 4 \
   --max_seq_len $max_seq_len \
   --learning_rate "9.65e-6" \
   --weight_decay 0. \
   --num_train_epochs $num_train_epochs  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim $lora_dim \
   --lora_module_name "layers." \
   --output_dir $OUTPUT \
   --print_loss \
   --save_interval 2000 \
   --enable_tensorboard \
   --tensorboard_path "${OUTPUT}/step1_tensorboard" \
   |& tee $OUTPUT/training.log