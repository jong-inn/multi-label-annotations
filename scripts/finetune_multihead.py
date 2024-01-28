import argparse
import numpy as np
import pandas as pd
import torch
import evaluate
from pathlib import Path
from torch import nn
from torch.nn import CrossEntropyLoss
from multi_headed_roberta import MultiHeadRobertaForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, Dataset, load_metric

DATASET_LIST = [
    "SChem5Labels",
    "Sentiment",
    "SBIC",
    "ghc"
]

DATASET_LABELS = {
    "SChem5Labels": ["0.0", "1.0", "2.0", "3.0", "4.0"],
    "Sentiment": ["0.0", "1.0", "2.0", "3.0", "4.0"],
    "SBIC": ["0.0", "1.0", "2.0"],
    "ghc": ["0.0", "1.0"]
}

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

class CustomValueDistanceLoss(nn.Module):
    def __init__(self):
        super(CustomValueDistanceLoss, self).__init__()

    def forward(self, y_true, y_pred):
        if y_true == -1 or y_pred == -1:
            return 0
        loss = 0.5 * torch.mean((y_true - y_pred)**2)
        return loss
val_dist_fct = CustomValueDistanceLoss()

class MultiHeadTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels").type(torch.FloatTensor).to(device)
        labels = inputs.pop("labels").type(torch.LongTensor).to(device)
        # print(f"labels: {labels}")
        # print(f"labels 0: {labels[:, 0]}")
        # print(f"labels size: {labels.size()}")
        logits_per_head = model(**inputs.to(device))
        # print(f"logits_per_head: {logits_per_head}")
        # print(f"logits_per_head size: {logits_per_head[0].size()}")
        loss_fct = CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        losses_per_head = [loss_fct(logits, labels[:, idx]) for idx, logits in enumerate(logits_per_head)]
        # print(f"losses_per_head: {losses_per_head}")
        total_loss = torch.stack(losses_per_head).mean()
        return (total_loss, logits_per_head) if return_outputs else total_loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     loss_fct = CrossEntropyLoss(ignore_index=-1)
    #     alpha = 0.8
        
    #     labels = inputs.pop("labels")
    #     outputs = model(**inputs.to(device))
    #     #print('*************************************')
    #     #for layer in model.module.classifiers.named_parameters():
    #     #    if 'weight' in layer[0]:
    #     #        print(torch.sum(layer[1]).item())
    #     loss = 0
    #     # sum up losses from all heads
    #     for i in range(len(labels)):
    #         for j in range(len(outputs)):
    #             ce = loss_fct(outputs[j][i], labels[i][j])
    #             pred_label = torch.argmax(outputs[j][i]).float()
    #             pred_label.requires_grad = True
    #             dist = val_dist_fct(labels[i][j], pred_label)
    #             #print("ce", ce, "dist", dist, "labels", labels[i][j], "pred_label", pred_label)
    #             loss += alpha * ce + (1-alpha) * dist 
    #     return (loss, outputs) if return_outputs else loss



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
    
    id2label = {idx: label for idx, label in enumerate(DATASET_LABELS[dataset])}
    label2id = {label: idx for idx, label in enumerate(DATASET_LABELS[dataset])}
    
    if human_model == "human":
        num_heads = HUMAN_NUM_ANNOTATORS[dataset]
    elif human_model == "model":
        num_heads = 5
    
    model = MultiHeadRobertaForSequenceClassification.from_pretrained(
        base_model, 
        device_map="auto",
        num_heads=num_heads,
        custom_num_labels=len(DATASET_LABELS[dataset]),
        # id2label=id2label,
        # label2id=label2id
    )
    model.to(device)
    
    if human_model == "human":
        output_dir = Path(__file__).parent.parent.joinpath("model", f"{dataset}_{human_model}_{base_model}_multihead_finetuned")
    elif human_model == "model":
        output_dir = Path(__file__).parent.parent.joinpath("model", f"{dataset}_{human_model}_{base_model}_multihead_finetuned")
    
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
        metric_for_best_model="f1"
    )
    
    accuracy = evaluate.load("f1")
    
    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return accuracy.compute(predictions=predictions, references=labels)
    
    def compute_metrics(eval_preds):
        print("INSIDE COMPUTE METRICS")
        metric = evaluate.load("f1")
        labels = eval_preds.label_ids.flatten()
        labels2 = eval_preds.label_ids
        logits = eval_preds.predictions
        predictions = np.argmax(logits, axis=0).flatten()
        if len(labels) != len(predictions):
            print("====len logits", len(logits), len(logits[0]), len(logits[0][0]))
            print("====len labels", len(labels2), labels2[0])
            print('labels', eval_preds.label_ids[0], 'predictions', np.argmax(logits, axis=0)[0])
            raise Exception("labels and predictions are not the same length")
        return metric.compute(predictions=predictions, references=labels, average="macro")
    
    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train_ds,
        eval_dataset=encoded_valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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
    
    if human_model == "human":
        target_column = "human_annots"
    elif human_model == "model":
        target_column = "model_majority"
    
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
