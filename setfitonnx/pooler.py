from torch import nn
import torch
from transformers import AutoModel


class MeanPooling(nn.Module):
    """ MeanPooling Module for Sentence Transformer
    """
    def __init__(self,path:str):
        super().__init__()
        self.model = AutoModel.from_pretrained(path)
        

    def forward(self,**inputs):
        tokens = self.model(**inputs)[0]
        mask = inputs["attention_mask"]
        mask = mask.unsqueeze(-1).expand(tokens.size()).float()
        return torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

class MeanPoolingOnnx(MeanPooling):
    """
    Extends MeanPooling class to name inputs to model, which is required
    to export to ONNX.
    """

    # pylint: disable=W0221
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Build list of arguments dynamically since some models take token_type_ids
        # and others don't
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        return super().forward(**inputs)