import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.Sign_Encoder import SignEncoder
from models.huggingface.modeling_xglm import XGLMForCausalLM
from transformers import XGLMTokenizer

from sacrebleu.metrics import BLEU

class FineTuneModel(pl.LightningModule):
    def __init__(self, 
                xglm_path="/home/sobhan/Documents/Code/xglm-1.7B", 
                tokenizer_path="/home/sobhan/Documents/Code/xglm-1.7B",
                lr=3e-4, 
                encoder_ckpt=None,
                ):
        super().__init__()
        self.save_hyperparameters()

        ################Set the Sign Encoder####################
        if encoder_ckpt is not None:
            # self.sign_encoder = SignEncoder(encoder_ckpt)
            pass
        else:
            self.sign_encoder = SignEncoder()
        #################Set the XGLM Model####################
        self.xglm = XGLMForCausalLM.from_pretrained(xglm_path)
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
        #################Initialize the tokenizer####################
        self.tokenizer = XGLMTokenizer.from_pretrained(tokenizer_path)
        #################Set the Projection####################
        self.proj = nn.Linear(512, 2048)
        #################Set the Optimizer####################
        self.lr = lr
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.validation_decoded = []
        self.validation_step_outputs = []
        
        self.test_decoded = []
        self.test_step_outputs = []

    def forward(self, list_of_frames, input_ids, attention_mask, max_len=1024):
        encoded_sign = self.sign_encoder(list_of_frames)
        adaptors = self.proj(encoded_sign["post_output"]['x'])
        masks = encoded_sign["post_output"]['mask']
        outputs = self.xglm(input_ids=input_ids, attention_mask=attention_mask, 
                            inputs_adaptors=adaptors, adaptor_mask=masks)
        return outputs['logits']

    def training_step(self, batch, batch_idx):
        logits = self(batch['list_of_frames'], batch['input_ids'], batch['attention_mask'])
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = self.calc_loss(logits, batch['labels'])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch['list_of_frames'], batch['input_ids'], batch['attention_mask'])
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = self.calc_loss(logits, batch['labels'])
        self.log_dict({
            "val_loss": loss,
            }, on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_decoded.extend(self.generate(batch['list_of_frames']))
        self.validation_step_outputs.append(batch['labels'])
        return loss

    def on_validation_epoch_end(self):
        max_length = max(tensor.size(1) for tensor in self.validation_step_outputs)
        padded_tensors = []
        for tensor in self.validation_step_outputs:
            # Calculate the number of padding values required for this tensor
            padding_size = max_length - tensor.size(1)
            
            # Pad tensor: (left_padding, right_padding) for sequence length (dimension 1)
            padded_tensor = F.pad(tensor, (0, padding_size), mode='constant', value=1)
            padded_tensors.append(padded_tensor)

        # Concatenate all the padded tensors into one tensor of shape (2 * t, max_length)
        targets = torch.cat(padded_tensors, dim=0)
        tgt_refs = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in targets]
        hypotheses = [item for item in self.validation_decoded]
        
        bleu = BLEU()
        bleu_s = bleu.corpus_score(hypotheses, [tgt_refs]).score
        
        self.log("val_bleu", bleu_s ,prog_bar=True)
        self.validation_decoded = []
        self.validation_step_outputs = []
    
    def test_step(self, batch, batch_idx):
        logits = self(batch['list_of_frames'], batch['input_ids'], batch['attention_mask'])
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = self.calc_loss(logits, batch['labels'])
        self.log("test_loss", loss)
        self.test_decoded.extend(self.generate(batch['list_of_frames']))
        self.test_step_outputs.append(batch['labels'])
        return loss
    
    def on_test_epoch_end(self):
        max_length = max(tensor.size(1) for tensor in self.test_step_outputs)
        padded_tensors = []
        for tensor in self.test_step_outputs:
            # Calculate the number of padding values required for this tensor
            padding_size = max_length - tensor.size(1)
            
            # Pad tensor: (left_padding, right_padding) for sequence length (dimension 1)
            padded_tensor = F.pad(tensor, (0, padding_size), mode='constant', value=1)
            padded_tensors.append(padded_tensor)

        # Concatenate all the padded tensors into one tensor of shape (2 * t, max_length)
        targets = torch.cat(padded_tensors, dim=0)
        tgt_refs = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in targets]
        hypotheses = [item for item in self.test_decoded]
        
        bleu = BLEU()
        bleu_s = bleu.corpus_score(hypotheses, [tgt_refs]).score
        
        self.log("test_bleu", bleu_s ,prog_bar=True)
        self.test_decoded = []
        self.test_step_outputs = []
    
    def generate(self, list_of_frames, max_len=55, num_beams=4):
        bsz = len(list_of_frames)
        # input_ids = [[self.tokenizer.bos_token_id] for _ in range(bsz)]
        # print(input_ids)
        input_ids = torch.zeros(bsz, 1, dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(self.device)
        
        encoded_sign = self.sign_encoder(list_of_frames)
        adaptors = self.proj(encoded_sign["post_output"]['x'])
        masks = encoded_sign["post_output"]['mask']
        outputs = self.xglm.generate(
                            input_ids=input_ids, attention_mask=attention_mask, 
                            inputs_adaptors=adaptors, adaptor_mask=masks, 
                            max_length=max_len, num_beams=num_beams,
                            )
        
        generated_texts = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
        return generated_texts
    
    def calc_loss(self, logits, y):
        return self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
    
    def configure_optimizers(self):

        print(f'lr: {self.lr}')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.05,  # 5% of total steps for warmup
                anneal_strategy='cos'
            ),
            "interval": "step",
            "frequency": 1,
        }
        
        return [optimizer], [scheduler]
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        # Implement your own custom logic to clip gradients
        # You can call `self.clip_gradients` with your settings:
        self.clip_gradients(
            optimizer,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )
        