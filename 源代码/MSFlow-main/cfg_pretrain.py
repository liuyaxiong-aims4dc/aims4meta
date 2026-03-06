import os
import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from data import  CondMolDataset
from modules.cond_lit_model import CondFlowMolBERTLitModule
from configs import data,lit_model
from pytorch_lightning.callbacks import EarlyStopping
from configs.data import TOK2ID, vocab_size
import torch
import numpy as np
local_rank = int(os.environ.get("LOCAL_RANK", 0))
print(f"Process {local_rank} using device: cuda:{local_rank}")


torch.set_float32_matmul_precision('medium')

def main():
    df_train= pd.read_parquet(data.training_data_path)
    df_val = pd.read_parquet(data.val_data_path)
    encoded_train = df["encoded"].apply(lambda x: x[:lit_model.MAX_LEN]).tolist()
    encoded_val= df_val["encoded"].apply(lambda x: x[:lit_model.MAX_LEN]).tolist()
    condition_train = df_train.cddds  #smiles column for ecfp
    condition_val = df_val.cddds
    train_dataset = CondMolDataset(encoded_train,condition_train)
    val_dataset = CondMolDataset(encoded_val,condition_val)
    print("Created datasets")
    train_loader = DataLoader(train_dataset, batch_size=data.batch_size, shuffle=True, num_workers=14, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=data.batch_size, shuffle=False, num_workers=14, pin_memory = True)

    wandb_base_dir = "wandb"
    run_id = None
    name = f'MSFlow_2.8M_LR={lit_model.lr}_{lit_model.source}_dim={lit_model.d_model+1}_cddd'
    wandb_logger = WandbLogger(
        project="morflow",
        name=f"{name}",
        save_dir=wandb_base_dir,
        resume="allow",
        id=run_id
    )

    cond_model= CondFlowMolBERTLitModule(
        model_name= lit_model.model_name,
        vocab_size=vocab_size,
        time_dim= 1,
        hidden_dim= lit_model.d_model,
        cond_dim=lit_model.COND_DIM,
        n_layers= lit_model.n_layers,
        n_heads= lit_model.n_heads,
        mlp = lit_model.mlp,
        max_len= lit_model.max_len,
        dropout= lit_model.dropout,
        uncond_prob=lit_model.uncond_prob,
        lr=lit_model.lr,
        warmup_ratio=lit_model.warmup_ratio,
        pad_token_id=TOK2ID[data.PAD],
        mask_token_id=TOK2ID[data.MASK],
        device=lit_model.device,
        source= lit_model.source,
        scheduler = lit_model.scheduler,
        path = lit_model.path,
        loss_fn = lit_model.loss,
        weighted=lit_model.weighted
    )
    
    early_stop_callback = EarlyStopping(
    monitor="val_loss",      
    patience=5,              
    mode="min",             
    verbose=True)

    checkpoint_dir = data.output_path
    last_ckpt_path = os.path.join(checkpoint_dir, "CFG_FM_last.ckpt")
    resume_ckpt = last_ckpt_path if os.path.exists(last_ckpt_path) else None


    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename= name + '_best_cond_val-{epoch:02d}-{cond_validity:.4f}',
        monitor="cond_validity",
        mode="max",
        save_top_k=1,
        save_last = True,
        save_weights_only=False,  # save full checkpoint, or True to save only model weights
    )


    trainer = Trainer(
        max_steps = lit_model.max_steps,
        accelerator="gpu",
        strategy="ddp",
        # precision=16,
        devices=4, 
        logger=[wandb_logger],
        callbacks=[ckpt_callback,early_stop_callback],
    )

    # --- Train ---
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"Rank {rank}, Local Rank {local_rank}, GPU {torch.cuda.current_device()}")
    if torch.distributed.is_initialized():
        print(f"[RANK {torch.distributed.get_rank()}] world_size={torch.distributed.get_world_size()} "
            f"local_rank={os.environ.get('LOCAL_RANK')} "
            f"cuda={torch.cuda.current_device()}")
    trainer.fit(cond_model, train_loader, val_loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
