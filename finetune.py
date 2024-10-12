import torch
import torch.backends.cudnn as cudnn
from models.Finetune_Model import FineTuneModel
from models.Sign_Encoder import get_sign_encoder
from models.spatial_models.frame_models.dino_adaptor_model import Model
from models.metaformer.meta_model import MetaFormer
from models.huggingface.modeling_xglm import XGLMForCausalLM

from dataset.slt_dataset import DataModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import argparse
from pathlib import Path
from datetime import datetime

torch.set_float32_matmul_precision("medium")


def get_args_parser():
    parser = argparse.ArgumentParser('Sign2GPT', add_help=False)
    
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--num_gpus', default=1, type=int, metavar='N', help='number of gpus per node')
    ##################Transformer and Encoder Params####################################    
    parser.add_argument('--xglm_path', type=str, default="/home/sobhan/Documents/Code/xglm-1.7B",
                        help='Path to the XGLM model.')
    parser.add_argument('--tokenizer_path', type=str, default="/home/sobhan/Documents/Code/xglm-1.7B",
                        help='Path to the XGLM tokenizer.')
    parser.add_argument('--encoder_ckpt', type=str, default=None, help='Path to the encoder checkpoint.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    ##################Data Params##########################################################
    parser.add_argument('--text_path', type=str, default="/home/sobhan/Documents/Code/LLaMA-Adapter/SQA-Lightning/src/sqa/data/labels", 
                        help='Path to the text data.')
    parser.add_argument('--data_config', type=str, default='configs/config.yaml',
                        help='Path to the data config file.')  
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--data_ver', type=int, default=0, help='Data version.')
    
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output_dir', type=str, default="output", help='Output directory.')
    parser.add_argument('--log_dir', type=str, default="output", help='Output directory.')
    return parser

def main(args):
    pl.seed_everything(args.seed)
    # fix the seed for reproducibility
    cudnn.benchmark = True

    # set logger
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logger = TensorBoardLogger(save_dir=f'{args.log_dir}/log_{current_time}', name="Sign2GPT")
    dirpath = f'{args.output_dir}/run_{current_time}'
    print("Current Time = {}".format(current_time)) 
    
    # set callbacks
    checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    save_last=True,
    monitor="val_loss",
    mode="min",
    dirpath=dirpath,
    filename="best-{epoch:03d}-{val_loss:.3f}",
    )
    early_stop = EarlyStopping("val_loss", patience=args.epochs, mode="min", verbose=True)
    callbacks = [checkpoint_callback, early_stop]
    
    model = FineTuneModel(
                xglm_path=args.xglm_path, 
                tokenizer_path=args.tokenizer_path,
                lr=args.lr, 
                encoder_ckpt=args.encoder_ckpt,)
    
    data_module = DataModule(
                root_text_path=args.text_path, 
                data_config=args.data_config,
                qa_csv_path=None,
                tokenizer_path=args.tokenizer_path,
                batch_size=args.batch_size, 
                num_workers=args.num_workers,
                data_ver=args.data_ver)
    
    trainer = pl.Trainer(
        logger=logger,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=list(range(args.num_gpus)),
        min_epochs=1,
        max_epochs=1,
        precision=16,
        callbacks=callbacks,
    )
    
    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    # # print(torch.version.cuda)
    # print(torch.cuda.is_available())
    # dummy_input = [torch.randn(4,3,224,224), torch.randn(5,3,224,224),
    #             torch.randn(6,3,224,224)]

    # model = FineTuneModel().to('cuda')
    # y = model(list_of_frames=dummy_input, input_ids=torch.tensor([[1,2,3], [1,2,3], [1,2,3]]).to('cuda'))
    # print(y.shape)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))


    # model_name, sign_model_params, dim_model = get_sign_encoder()
    # spatial_params = sign_model_params['spatial_params']
    # model = Model(**spatial_params, out_dim=dim_model).to('cuda')
    # dummy_input = [torch.randn(4,3,224,224), torch.randn(5,3,224,224),
    #             torch.randn(6,3,224,224)]
    # y, mask,_ = model(dummy_input)

    # print(f'Spatial Model info')
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'The number of trainable parameters in {model_name} is = {num_params}.')
    # print(y.shape)
    # print(mask.shape)

    # print('*'*10)
    # # print(mask)


    # encoder_name = sign_model_params['encoder_name']
    # encoder_params = sign_model_params['encoder_params']
    # print(encoder_params)
    # encoder = MetaFormer(**encoder_params).to('cuda')
    # # print(y.device, mask.device)
    # outputs = encoder(y, mask)
    # post_output = outputs["post_output"]['x']
    # hidden_state = outputs["hidden_state"]
    # hidden_mask = outputs["hidden_mask"]
    # print(f'Encoder Model info')
    # num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    # print(f'The number of trainable parameters in {encoder_name} is = {num_params}.')
    # print(post_output.shape)
    # print(hidden_state.shape)
    # print(hidden_mask.shape)
    # print('*'*10)
    # additional_tokens = {
    #     "eos_token": ".",
    # }
    # pretext = ""
    # new_token_length = None
    # import numpy as np
    # # lang_model = XGLMForCausalLM.from_pretrained(llm_name)
    # adaptor_params = {
    #     "adapt_layers": list(np.arange(0, 24, 1)),
    #     "lora_layers": list(np.arange(0, 24, 1)),
    #     "w_lora_ff": False,
    #     "lora_rank": 4,
    #     "lora_drop": 0.1,
    #     "gate_type": "clamp",
    #     "lora_a": 4.0,
    #     "adapt_tokens": False,
    # }
    # lang_model = XGLMForCausalLM.from_pretrained("facebook/xglm-1.7B")
    # if new_token_length is not None:
    #     lang_model.resize_token_embeddings(new_token_length)
    # for name, param in lang_model.named_parameters():
    #     param.requires_grad = False
    # lang_model.init_adaptor(**adaptor_params)
    
    # lang_model = lang_model.to('cuda')
    # torch.nn.Linear(512,1024).cuda()(outputs["post_output"]['x'])
    # output_lang = lang_model(input_ids=torch.tensor([[1,2,3], [1,2,3], [1,2,3]]).to('cuda'), inputs_adaptors=torch.nn.Linear(512,2048).cuda()(outputs["post_output"]['x']), adaptor_mask=outputs["post_output"]['mask'])
    # print(output_lang['logits'].shape)
    # lang_model.generate(input_ids=torch.tensor([[1,2,3], [1,2,3], [1,2,3]]).to('cuda'))
