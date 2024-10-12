import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.Sign_Encoder import SignEncoder
from models.huggingface.modeling_xglm import XGLMForCausalLM

class FineTuneModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.sign_encoder = SignEncoder()
        self.xglm = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")

        new_token_length = None
        if new_token_length is not None:
            self.xglm.resize_token_embeddings(new_token_length)
        for name, param in self.xglm.named_parameters():
            param.requires_grad = False
        
        adaptor_params = {
        "adapt_layers": list(np.arange(0, 24, 1)),
        "lora_layers": list(np.arange(0, 24, 1)),
        "w_lora_ff": False,
        "lora_rank": 4,
        "lora_drop": 0.1,
        "gate_type": "clamp",
        "lora_a": 4.0,
        "adapt_tokens": False,
        }
        self.xglm.init_adaptor(**adaptor_params)
        self.proj = nn.Linear(512, 2048)

    def forward(self, x):
        encoded_sign = self.sign_encoder(x)
        adaptors = self.proj(encoded_sign["post_output"]['x'])
        masks = encoded_sign["post_output"]['mask']
        outputs = self.xglm(input_ids=outputs, inputs_adaptors=adaptors, adaptor_mask=masks)

        return outputs['logits']

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = self.calc_loss(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = self.calc_loss(logits, y)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = self.calc_loss(logits, y)
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
    
    def calc_loss(self, logits, y):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        