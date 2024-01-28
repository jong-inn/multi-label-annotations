import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from utils import DATASET_LABELS
from labels_dist_roberta import MultiHeadRobertaForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

DATASET_LIST = [
    "SChem5Labels",
    "Sentiment",
    "SBIC",
    "ghc"
]

MODEL_ANNOTATOR_IDX = {
    0: "vicuna",
    1: "baize",
    2: "llama2",
    3: "koala",
    4: "open_ai_gpt35turbo"
}

HUMAN_NUM_ANNOTATORS = {
    "SChem5Labels": 5,
    "Sentiment"   : 4,
    "SBIC"        : 3,
    "ghc"         : 3
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

def main(
    dataset: str,
    base_model: str,
    human_model: str,
    sum_constraint: bool
):
    print(f"sum_constraint: {sum_constraint}")
    test_df = load_dataset_custom(dataset, human_model)
    
    model_path = Path(__file__).parent.parent.joinpath("model", f"{dataset}_{human_model}_{base_model}_labels_dist_finetuned")
    folder_list = [p for p in model_path.iterdir() if p.is_dir()]
    folder_keys = [int(p.name.replace("checkpoint-", "")) for p in folder_list]
    folder_keys.sort()
    print(f"folder_keys: {folder_keys}")
    last_checkpoint = model_path.joinpath(f"checkpoint-{folder_keys[-1]}")
    print(f"last_checkpoint: {last_checkpoint}")
    model_path = model_path.joinpath(last_checkpoint)
    save_path = model_path.parent.parent.parent.joinpath("prediction", f"{dataset}_{human_model}_{base_model}_{sum_constraint}_labels_dist_prediction.csv")
    
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    
    num_heads = DATASET_LABELS[dataset]
    
    model = MultiHeadRobertaForSequenceClassification.from_pretrained(
        model_path, 
        device_map="auto",
        custom_num_labels=len(DATASET_LABELS[dataset])
    )
    model.to(device)
    
    
    def apply_constraint(predictions: list, sum_constraint: bool, sum_labels: int):
        # Apply softmax to convert predictions to probabilities
        probabilities = [F.softmax(prediction, dim=1) for prediction in predictions]
        
        # Get a sample count using probabilities until meet the sum constraint
        if sum_constraint:
            while True:
                # counts = [torch.multinomial(probability, num_samples=1).squeeze().item() for probability in probabilities]
                count_weights = [max(probability) for probability in probabilities]
                counts = torch.multinomial(count_weights, num_samples=5)
                if sum(counts) == sum_labels:
                    break
        else:
            count_weights = [max(probability) for probability in probabilities]
            counts = torch.multinomial(count_weights, num_samples=5)
            # counts = [torch.multinomial(probability, num_samples=1).squeeze().item() for probability in probabilities]

        return counts.tolist()
    
    if human_model == "model":
        sum_labels = 5
    elif human_model == "human":
        sum_labels = HUMAN_NUM_ANNOTATORS[dataset]
    
    predicts = []
    with torch.no_grad():
        for prompt in test_df["prompt"]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            
            predicted = apply_constraint(outputs, sum_constraint, sum_labels)
            
            # print(f"output: {model(**inputs)}")
            # predicted = [output.argmax().item() for output in model(**inputs)]
            # logits = [output.argmax().item() for output in model(**inputs)]
            # print(f"logits: {logits}")
            # predicted = [logit.argmax().item() for logit in logits]
            predicts.append(predicted)
    # count = Counter(predicts)
    # print(f"count: {count}")
    
    test_df["prediction"] = predicts
    test_df.to_csv(save_path, index=False)
    
def load_dataset_custom(dataset: str, human_model: str) -> pd.DataFrame:
    data_path = Path(__file__).parent.parent.joinpath("data")
    
    test_df = pd.read_csv(data_path.joinpath(f"df_final_{dataset}_test.csv"))

    string_to_int = {
        "'0.0'": 0,
        "'1.0'": 1,
        "'2.0'": 2,
        "'3.0'": 3,
        "'4.0'": 4
    }
    
    test_df["human_annots"] = test_df["human_annots"].str.strip("[]").str.split(",").apply(lambda x: [string_to_int[i.strip()] for i in x])
    test_df["model_majority"] = test_df["model_majority"].str.strip("[]").str.split(",").apply(lambda x: [string_to_int[i.strip()] for i in x])
    
    def annotation_to_count(annotations):
        count = {label: 0 for label in DATASET_LABELS[dataset]}
        for annotation in annotations:
            count[annotation] += 1
        return list(count.values())
    
    test_df["human_label_count"] = test_df["human_annots"].apply(lambda x: annotation_to_count(x))
    test_df["model_label_count"] = test_df["model_majority"].apply(lambda x: annotation_to_count(x))
    
    test_df["idx"] = test_df.index
    test_df = test_df[["idx", "prompt", "human_annots", "model_majority", "human_label_count", "model_label_count"]]
    
    print(f"""
dataset: {dataset}, {human_model}
shape: {test_df.shape}
head:
    {test_df.head()}
          """)
    return test_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="SChem5Labels",
        choices=DATASET_LIST
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="roberta-large"
    )
    parser.add_argument(
        "--human_model",
        type=str,
        default="model",
        choices=["human", "model"]
    )
    parser.add_argument(
        "--sum_constraint",
        action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()
    if not args.sum_constraint:
        sum_constraint = False
    else:
        sum_constraint = True
    
    main(
        args.dataset,
        args.base_model,
        args.human_model,
        sum_constraint
    )