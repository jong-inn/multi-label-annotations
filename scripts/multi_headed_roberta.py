import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import RobertaModel, RobertaPreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput



class MultiHeadedClassifier(nn.Module):
    def __init__(self, config, num_heads, custom_num_labels):
        super(MultiHeadedClassifier, self).__init__()
        self.classifiers = nn.ModuleList([RobertaClassificationHead(config, custom_num_labels) for _ in range(num_heads)])

    def forward(self, sequence_output):
        # Forward pass through each classifier
        # logits_per_head = [head(outputs.last_hidden_state[:, 0, :]) for head in self.heads]
        logits_per_head = [classifier(sequence_output) for classifier in self.classifiers]
        return logits_per_head



class MultiHeadRobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config, num_heads, custom_num_labels):
        super(MultiHeadRobertaForSequenceClassification, self).__init__(config)
        # self.num_labels = config.num_labels
        # self.num_labels = self.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = MultiHeadedClassifier(config, num_heads, custom_num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # sequence_output = outputs[0]
            # logits = self.classifier(sequence_output)
            
            sequence_output = outputs[0]
            logits_per_head = self.classifier(sequence_output)
            
            return logits_per_head

            # loss = None
            # if labels is not None:
            #     # move labels to correct device to enable model parallelism
            #     labels = labels.to(logits.device)
            #     if self.config.problem_type is None:
            #         if self.num_labels == 1:
            #             self.config.problem_type = "regression"
            #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            #             self.config.problem_type = "single_label_classification"
            #         else:
            #             self.config.problem_type = "multi_label_classification"

            #     if self.config.problem_type == "regression":
            #         loss_fct = MSELoss()
            #         if self.num_labels == 1:
            #             loss = loss_fct(logits.squeeze(), labels.squeeze())
            #         else:
            #             loss = loss_fct(logits, labels)
            #     elif self.config.problem_type == "single_label_classification":
            #         loss_fct = CrossEntropyLoss()
            #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            #     elif self.config.problem_type == "multi_label_classification":
            #         loss_fct = BCEWithLogitsLoss()
            #         loss = loss_fct(logits, labels)
                    
            # if labels is not None:
            #     loss_fct = CrossEntropyLoss()
            #     losses = [loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) for logits in logits_per_head]
            #     total_loss = sum(losses)
                # losses = [loss_fct(logits, labels) for logits in logits_per_head]
                # total_loss = sum(losses)

            # if not return_dict:
            #     output = (logits,) + outputs[2:]
            #     return ((loss,) + output) if loss is not None else output

            # return SequenceClassifierOutput(
            #     loss=loss,
            #     logits=logits,
            #     hidden_states=outputs.hidden_states,
            #     attentions=outputs.attentions,
            # )



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, custom_num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, custom_num_labels)
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# Example: MultiHeadRobertaForSequenceClassification from the previous code
# ...

# Customized Trainer for multi-head and multi-label
# class MultiHeadTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         logits_per_head = model(**inputs)
#         losses_per_head = [nn.CrossEntropyLoss()(logits, labels) for logits in logits_per_head]
#         total_loss = torch.stack(losses_per_head).mean()
#         return (total_loss, logits_per_head) if return_outputs else total_loss

# Example: Prepare data, tokenizer, model, etc.
# ...

# Define TrainingArguments
# training_args = TrainingArguments(
#     output_dir="./output",
#     per_device_train_batch_size=8,
#     num_train_epochs=3,
#     # ... other training arguments
# )

# # Initialize MultiHeadTrainer
# trainer = MultiHeadTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,  # replace with your actual training dataset
#     # ... other Trainer arguments
# )

# # Train the model
# trainer.train()
