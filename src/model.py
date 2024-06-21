import torch
from torch import nn
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_classes_task_a, num_classes_task_b):
        super(MultiTaskModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier_task_a = nn.Linear(self.encoder.config.hidden_size, num_classes_task_a)
        self.classifier_task_b = nn.Linear(self.encoder.config.hidden_size, num_classes_task_b)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        logits_task_a = self.classifier_task_a(pooled_output)
        logits_task_b = self.classifier_task_b(pooled_output)
        return logits_task_a, logits_task_b
