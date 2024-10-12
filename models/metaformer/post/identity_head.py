import torch
import torch.nn as nn
# from models.model_utils.masked_norm import MaskedNorm


class Model_Post(nn.Module):
    def __init__(self, d_model=None, out_dim=None):
        super().__init__()

    def forward(self, x, mask):
        return {"x": x, "mask": mask}
