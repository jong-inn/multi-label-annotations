# CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_multihead.py --dataset=SChem5Labels --base_model="roberta-large" --human_model=human
CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_multihead.py --dataset=Sentiment --base_model="roberta-large" --human_model=human
# CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_multihead.py --dataset=SBIC --base_model="roberta-large" --human_model=human