import argparse
import numpy as np
import pandas as pd
import torch
import evaluate
from pathlib import Path
import torch.nn.functional as F
from torch.nn import MSELoss
from utils import DATASET_LABELS
from sklearn.metrics import mean_squared_error
from labels_dist_regression_roberta import MultiHeadRobertaForSequenceClassification
from transformers import RobertaTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset

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

# The number of human annotators
# SChem5Labels: 5
# Sentiment   : 4
# SBIC        : 3
# ghc         : 3


MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 2e-5 # 1e-05


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


class MultiHeadTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        targets_batch = inputs.pop("labels").type(torch.LongTensor).to(device)
        # print(f"targets_batch: {targets_batch}")
        
        predictions_per_head = model(**inputs.to(device))
        # print(f"predictions_per_head length: {len(predictions_per_head)}")
        # print(f"predictions_per_head 0 size: {predictions_per_head[0].size()}")
        # print(f"predictions_per_head 0: {predictions_per_head[0]}")
        
        losses = []
        for batch_idx in range(targets_batch.size()[0]):
            targets = targets_batch[batch_idx, :]
            predictions = torch.Tensor(list(map(lambda x: x[batch_idx], predictions_per_head))).to(device)
            loss_fct = MSELoss()
            loss = loss_fct(predictions, targets)
            loss.requires_grad_()
            loss.backward()
            losses.append(loss)
        total_loss = sum(losses)
        
        return (total_loss, predictions_per_head) if return_outputs else total_loss


def main(
    dataset: str,
    base_model: str,
    human_model: str
):
    
    train_ds, valid_ds = load_dataset_custom(dataset, human_model)
    
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def preprocess_function(examples):
        return tokenizer(examples["prompt"], truncation=True)
    
    encoded_train_ds = train_ds.map(preprocess_function, batched=True)
    encoded_valid_ds = valid_ds.map(preprocess_function, batched=True)
    
    model = MultiHeadRobertaForSequenceClassification.from_pretrained(
        base_model, 
        # device_map="auto",
        custom_num_labels=len(DATASET_LABELS[dataset]),
    )
    model.to(device)
    
    if human_model == "human":
        output_dir = Path(__file__).parent.parent.joinpath("model", f"{dataset}_{human_model}_{base_model}_labels_dist_regression_finetuned")
    elif human_model == "model":
        output_dir = Path(__file__).parent.parent.joinpath("model", f"{dataset}_{human_model}_{base_model}_labels_dist_regression_finetuned")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_steps=0.5,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        # metric_for_best_model="accuracy"
    )
    
    # def compute_metrics(eval_preds):
    #     print("INSIDE COMPUTE METRICS")
    #     mse_metrics = evaluate.load("mse")
    #     predictions_per_head = eval_preds.predictions
    #     labels = eval_preds.label_ids
    #     print(f"labels type: {type(labels)}")
    #     print(f"labels size: {len(labels)}")
    #     print(f"labels 0: {labels[0]}")
    #     print(f"labels 0 size: {labels[0].shape}")
    #     print(f"predictions_per_head type: {type(predictions_per_head)}")
    #     print(f"predictions_per_head size: {len(predictions_per_head)}")
    #     print(f"predictions_per_head 0 size: {predictions_per_head[0].shape}")
        
    #     total_mse = 0
    #     for batch_idx in range(len(labels)):
    #         targets = labels[batch_idx]
    #         predictions = list(map(lambda x: x[batch_idx].squeeze().item(), predictions_per_head))
    #         print(f"targets: {targets}")
    #         print(f"predictions: {predictions}")
    #         mse = mse_metrics.compute(predictions=predictions, references=targets)
    #         total_mse += mse["mse"]
            
    #     return {"mse": total_mse}
    
    def compute_metrics(eval_preds):
        print("INSIDE COMPUTE METRICS")
        accuracy = evaluate.load("accuracy")
        logits_per_head = eval_preds.predictions
        print(f"logits_per_head length: {len(logits_per_head)}")
        print(f"logits_per_head length 0 type: {type(logits_per_head[0])}")
        labels = eval_preds.label_ids
        
        accuracy_sum = 0
        for idx, logits in enumerate(logits_per_head):
            logits = torch.from_numpy(logits)
            probabilities = F.softmax(logits, dim=1)
            head_predictions = torch.argmax(probabilities, dim=1)
            print(f"head_predictions size: {head_predictions.size()}")
            head_predictions = [t.item() for t in head_predictions]
            
            sub_result = accuracy.compute(predictions=head_predictions, references=labels[:, idx])
            accuracy_sum += sub_result["accuracy"]
            
        return {"accuracy": accuracy_sum / len(logits_per_head)}
    

    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train_ds,
        eval_dataset=encoded_valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    pd.DataFrame(trainer.state.log_history).to_json(output_dir.joinpath("log_history.json"))
    trainer.evaluate()
    

def load_dataset_custom(dataset: str, human_model: str) -> Dataset:
    data_path = Path(__file__).parent.parent.joinpath("data")
    tmp_train = pd.read_csv(data_path.joinpath("df_final_train.csv"))
    tmp_valid = pd.read_csv(data_path.joinpath("df_final_valid.csv"))
    tmp_test = pd.read_csv(data_path.joinpath("df_final_test.csv"))
    
    for d in DATASET_LIST:
        if not data_path.joinpath(f"df_final_{d}_train.csv").exists():
            tmp_train.loc[tmp_train["dataset_name"] == d].to_csv(data_path.joinpath(f"df_final_{d}_train.csv"), index=False)
        if not data_path.joinpath(f"df_final_{d}_valid.csv").exists():
            tmp_valid.loc[tmp_valid["dataset_name"] == d].to_csv(data_path.joinpath(f"df_final_{d}_valid.csv"), index=False)
        if not data_path.joinpath(f"df_final_{d}_test.csv").exists():
            tmp_test.loc[tmp_test["dataset_name"] == d].to_csv(data_path.joinpath(f"df_final_{d}_test.csv"), index=False)
    
    train_df = pd.read_csv(data_path.joinpath(f"df_final_{dataset}_train.csv"))
    valid_df = pd.read_csv(data_path.joinpath(f"df_final_{dataset}_valid.csv"))

    string_to_int = {
        "0.0": 0,
        "1.0": 1,
        "2.0": 2,
        "3.0": 3,
        "4.0": 4
    }
    
    train_df["human_annots"] = train_df["human_annots"].str.strip("[]").str.replace("'", "").str.split(",").apply(lambda x: [string_to_int[i.strip()] for i in x])
    train_df["model_majority"] = train_df["model_majority"].str.strip("[]").str.replace("'", "").str.split(",").apply(lambda x: [string_to_int[i.strip()] for i in x])
    valid_df["human_annots"] = valid_df["human_annots"].str.strip("[]").str.replace("'", "").str.split(",").apply(lambda x: [string_to_int[i.strip()] for i in x])
    valid_df["model_majority"] = valid_df["model_majority"].str.strip("[]").str.replace("'", "").str.split(",").apply(lambda x: [string_to_int[i.strip()] for i in x])
    
    def annotation_to_count(annotations):
        count = {label: 0 for label in DATASET_LABELS[dataset]}
        for annotation in annotations:
            count[annotation] += 1
        return list(count.values())
    
    train_df["human_label_count"] = train_df["human_annots"].apply(lambda x: annotation_to_count(x))
    train_df["model_label_count"] = train_df["model_majority"].apply(lambda x: annotation_to_count(x))
    valid_df["human_label_count"] = valid_df["human_annots"].apply(lambda x: annotation_to_count(x))
    valid_df["model_label_count"] = valid_df["model_majority"].apply(lambda x: annotation_to_count(x))
    
    if human_model == "human":
        target_column = "human_label_count"
    elif human_model == "model":
        target_column = "model_label_count"
    
    train_df["idx"] = train_df.index
    train_df["label"] = train_df[target_column]
    train_df = train_df[["idx", "label", "prompt"]]
    
    valid_df["idx"] = valid_df.index
    valid_df["label"] = valid_df[target_column]
    valid_df = valid_df[["idx", "label", "prompt"]]
    
    train_ds = Dataset.from_pandas(train_df)
    valid_ds = Dataset.from_pandas(valid_df)
    
    print(f"""
train_ds: {train_ds}
valid_ds: {valid_ds}
          """)
    return train_ds, valid_ds


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
    
    args = parser.parse_args()
    
    main(
        args.dataset,
        args.base_model,
        args.human_model
    )
