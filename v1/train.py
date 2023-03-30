import torch
import torch.nn as nn
import time
import datetime
import subprocess 
import pytorch_lightning as pl
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import os
import numpy as np

def compute_correlations(labels, preds, return_detail = False):
    device = labels.device
    
    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    corr = np.nan_to_num([np.corrcoef(labels[:,i], preds[:,i])[0,1] for i in range(labels.shape[1])], nan = -1).tolist()
    if return_detail:
        return corr
    corr = np.mean(corr)
    return torch.FloatTensor([corr]).to(device)

def to_gene(gene_corr):
    output = ""
    for i,j in gene_corr:
        output += "[%s : %.4f] " % (i,j)
    return output

def pearsonr(x, y):

    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / (r_den + 1e-8)
    r_val = torch.nan_to_num(r_val,nan=-1)
    return r_val


class TrainerModel(pl.LightningModule):
    
    def __init__(self, config,  model):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.automatic_optimization = False
        self.min_loss  = float("inf")
        self.max_corr  = float("-inf")
        self.max_eval_corr = float("-inf")
        self.min_eval_loss = float("inf")
        self.start_time  = None
        self.last_saved = None
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset =  self.trainer._data_connector._train_dataloader_source.dataloader() 
        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes) * self.trainer.num_nodes
        return len(dataset) // num_devices
    
    def correlationMetric(self,x, y):
      corr = 0
      for idx in range(x.size(1)):
          corr += pearsonr(x[:,idx], y[:,idx])
      corr /= (idx + 1)
      return (1 - corr).mean()
    def training_step(self,data,idx):
        
        if self.current_epoch == 0 and idx == 0:
            self.start_time  = time.time()
        
        optimizer = self.optimizers()
        
        pred_count = self.model(data)
        loss   = self.criterion(pred_count,data["count"])
        corrloss = self.correlationMetric(pred_count,data["count"])
        
        optimizer.zero_grad()
        self.manual_backward(loss + corrloss * 0.5)
        optimizer.step()
        
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        
        self.produce_log(loss.detach(),corrloss.detach(),data,idx)
        
        
    def produce_log(self,loss,corr,data,idx):
        
        train_loss = self.all_gather(loss).mean().item()
        train_corr = self.all_gather(corr).mean().item()
        
        self.min_loss   = min(self.min_loss, train_loss)
        
        if self.trainer.is_global_zero and loss.device.index == 0 and idx % self.config.verbose_step == 0:
            
            current_lr = self.optimizers().param_groups[0]['lr']
            
            len_loader = self.num_training_steps
            
            batches_done = self.current_epoch  * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            time_left    = datetime.timedelta(seconds = batches_left * (time.time() - self.start_time) / batches_done)
                    
            self.config.logfun(
                        "[Epoch %d/%d] [Batch %d/%d] [Loss: %f, Corr: %f, lr: %f] [Min Loss: %f] ETA: %s" % 
                        (self.current_epoch,
                         self.trainer.max_epochs,
                         idx,
                         len_loader,
                         train_loss,
                         train_corr,
                         current_lr,
                         self.min_loss,
                         time_left
                            )
                        
                        )
            

        
    def validation_step(self,data,idx):
        pred_count = self.model(data)
        return pred_count,data["count"]
        
    def validation_epoch_end(self,outputs):
        
        logfun = self.config.logfun
        
        pred_count = torch.stack([i[0] for i in outputs])
        count = torch.stack([i[1] for i in outputs])
        pred_count = self.all_gather(pred_count).view(-1,250)
        count = self.all_gather(count).view(-1,250)
        
        total_loss = self.criterion(pred_count,count).item()
        gene_corr = compute_correlations(count, pred_count, True)
        corr = np.mean(gene_corr)
        gene_corr = sorted(list(zip(self.config.filter_name, gene_corr)), key = lambda x : x[1])
        
        
        if self.trainer.is_global_zero and self.trainer.num_gpus != 0:
            for line in subprocess.check_output(["nvidia-smi"]).decode("utf-8").split("\n"):
                self.config.logfun(line)
            
            if corr > self.max_eval_corr:
                 self.save(self.current_epoch, total_loss,corr)                    
            self.max_eval_corr = max(self.max_eval_corr,corr)
            self.min_eval_loss = min(self.min_eval_loss, total_loss)
                            
            logfun("==" * 25)
            logfun(
                "[Corr :%f, Loss: %f] [Min Loss :%f, Max Corr: %f]" %
                (corr,
                 total_loss,
                 self.min_eval_loss,
                 self.max_eval_corr,
                 )
                )            
            logfun("Top 10 gene corr")
            logfun(to_gene(gene_corr[-10:][::-1]))
            logfun("==" * 25)
            logfun("End Evaluation")
        
    def save(self, epoch,loss, acc):
        
        self.config.logfun(self.last_saved)
        if self.last_saved != None:
            os.remove(self.last_saved)
        output_path = os.path.join(self.config.store_dir, "%d_%f_%f.pt" % (epoch,loss,acc))
        self.last_saved = output_path
        torch.save(self.model.state_dict(), output_path)
        self.config.logfun("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
                      
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
                            self.parameters(),
                            lr = self.config.lr,
                            betas = (0.9, 0.999),
                            weight_decay = self.config.weight_decay,
            )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.num_training_steps//2,
            cycle_mult=1.0,
            max_lr=self.config.lr,
            min_lr=self.config.lr/5,
            warmup_steps= 100
        )
        
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}                       
                      
                        
                        
                        
        
     
