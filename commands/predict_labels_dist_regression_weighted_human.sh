CUDA_VISIBLE_DEVICES=0 python ../scripts/predict_labels_dist_regression_weighted.py --dataset=SChem5Labels --base_model="roberta-large" --human_model=human
CUDA_VISIBLE_DEVICES=0 python ../scripts/predict_labels_dist_regression_weighted.py --dataset=Sentiment --base_model="roberta-large" --human_model=human