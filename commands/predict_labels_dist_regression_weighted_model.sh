CUDA_VISIBLE_DEVICES=1 python ../scripts/predict_labels_dist_regression_weighted.py --dataset=SChem5Labels --base_model="roberta-large" --human_model=model
CUDA_VISIBLE_DEVICES=1 python ../scripts/predict_labels_dist_regression_weighted.py --dataset=Sentiment --base_model="roberta-large" --human_model=model