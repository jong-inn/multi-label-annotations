from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput



class MultiHeadedClassifier(nn.Module):
    def __init__(self, config, custom_num_labels):
        super(MultiHeadedClassifier, self).__init__()
        # Define regression heads
        self.regression_heads = nn.ModuleList([
            RobertaClassificationHead(config, 1) for _ in range(custom_num_labels)
        ])
        
        # self.regression_heads = nn.ModuleList([
        #     nn.Linear(config.hidden_size, 1) for _ in range(custom_num_labels)
        # ])

    def forward(self, sequence_output):
        # Forward pass through each classifier
        predictions_per_head = [head(sequence_output) for head in self.regression_heads]
        return predictions_per_head


class MultiHeadRobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config, custom_num_labels):
        super(MultiHeadRobertaForSequenceClassification, self).__init__(config)
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = MultiHeadedClassifier(config, custom_num_labels)
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
            
            sequence_output = outputs[0]
            predictions_per_head = self.classifier(sequence_output)
            
            return predictions_per_head


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, output_size):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, output_size)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x