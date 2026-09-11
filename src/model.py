import torch.nn as nn
from transformers import AutoModel


class BertMultiLabelClassifier(nn.Module):
    """CLS-pooled BERT/DistilBERT encoder + linear multi-label classification head."""

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        # "eager" attention is required so output_attentions=True actually returns
        # attention weights -- transformers' default "sdpa" backend does not support it.
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, output_attentions: bool = False):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        cls_hidden = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_hidden)
        if output_attentions:
            return logits, outputs.attentions
        return logits
