import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from utils.metrics import decode_tokens_to_smiles, compute_smiles_metrics
from utils.sample import  cond_generate_mols
from configs.data import MAX_LEN
from configs import *
from models.cfg_molbert import CondFlowMolBERT
from trainers import  cond_dfm

class CondFlowMolBERTLitModule(pl.LightningModule):
    def __init__(
        self,
        model_name='dfm',  
        vocab_size=173,
        time_dim = 1,
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        mlp=256,
        cond_dim = 1449,
        max_len = 72,
        dropout= 0.4,
        uncond_prob = 0.1,
        lr=1e-3,
        warmup_ratio=0.1,
        pad_token_id=1,
        mask_token_id=0,
        device='cuda',
        source = 'masked', # uniform
        scheduler = PolynomialConvexScheduler(n=1.0),
        path =  MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0)),
        loss_fn = nn.CrossEntropyLoss(),
        weighted = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.model = CondFlowMolBERT(vocab_size,cond_dim,time_dim, hidden_dim, n_layers, n_heads, mlp,max_len=max_len,dropout=dropout)
        self.lr_scheduler = None  # Initialized in on_fit_start
        self.automatic_optimization = False
        self.first_val_batch = None

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr,betas=(0.9, 0.95),weight_decay=0.01)
        return optimizer


    def on_fit_start(self):
        max_steps = self.trainer.max_steps
        warmup_steps = int(self.hparams.warmup_ratio * max_steps)

        optimizer = self.optimizers()
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            num_cycles=0.5
        )

    def training_step(self, batch):
        optimizer = self.optimizers()
        loss = None
        if self.model_name == 'dfm':
            loss = cond_dfm.dfm_step(
                batch[0], batch[1], self.model,self.hparams.source,self.hparams.loss_fn,self.hparams.scheduler, self.hparams.path, 
                self.hparams.device,
                self.hparams.mask_token_id,
                self.hparams.weighted, self.hparams.uncond_prob, training = True
            )
            self.log("train_loss", loss.item(), prog_bar=True)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")
        
        self.manual_backward(loss)
        ##added clipping for preventing exploding gradients 
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        self.lr_scheduler.step()
        optimizer.zero_grad()
        self.log("lr", optimizer.param_groups[0]["lr"], prog_bar=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch,batch_idx):
        # if batch_idx == 0:  
        #     self.first_val_batch = batch
        loss = cond_dfm.dfm_step(
                batch[0],batch[1] ,self.model,self.hparams.source,self.hparams.loss_fn,self.hparams.scheduler, self.hparams.path, 
                self.hparams.device,
                self.hparams.mask_token_id,
                self.hparams.weighted, self.hparams.uncond_prob

            )
        print("validation step running")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        return loss
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        val_loader = self.trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))
        samples = cond_generate_mols(self.model, batch[1], num_samples=len(batch[1]))
        total_samples = len(samples)
        _, smiles = decode_tokens_to_smiles(samples, ID2TOK=ID2TOK, TOK2ID=TOK2ID, PAD=PAD)
        metrics = compute_smiles_metrics(total_samples=total_samples, decoded_smiles=smiles)
        self.log("cond_validity", metrics['validity'], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("cond_uniqueness", metrics['uniqueness'], on_epoch=True, sync_dist=True)
        self.log("cond_diversity", metrics['diversity'], on_epoch=True, sync_dist=True)
