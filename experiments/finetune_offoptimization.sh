export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ..
python3 scripts/finetune.py \
--seq_len 1024 \
--bit_precision -1 \
--enable_lora 0