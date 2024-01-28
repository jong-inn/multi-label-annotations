CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_labels_dist.py --dataset=SChem5Labels --base_model="roberta-large" --human_model=human
# CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_labels_dist.py --dataset=Sentiment --base_model="roberta-large" --human_model=human
# CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_labels_dist.py --dataset=SBIC --base_model="roberta-large" --human_model=human
# CUDA_VISIBLE_DEVICES=0 python ../scripts/finetune_labels_dist.py --dataset=ghc --base_model="roberta-large" --human_model=human