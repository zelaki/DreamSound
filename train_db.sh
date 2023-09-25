export CUDA_VISIBLE_DEVICES=2
export MODEL_NAME="audioldm2-music"
export DATA_DIR="/home/plitsis/text_inv/audio_inv/concepts/oud"
export OUTPUT_DIR="oud_ldm2_db_trumpet"
let base_port=29600+$CUDA_VISIBLE_DEVICES
accelerate launch --num_processes 1 --main_process_port $base_port dreambooth_audioldm2.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$DATA_DIR \
--instance_word="sks" \
--object_class="trumpet" \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--max_train_steps=300 \
--learning_rate=1.0e-05 \
--output_dir=$OUTPUT_DIR \
--validation_steps=50 \
--num_validation_audio_files=3 \
--num_vectors=1 