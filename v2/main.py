import argparse
import os
from model import HeteroGNN
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from train import TrainerModel
from pytorch_lightning.plugins import DDPPlugin
import glob
cudnn.benchmark = True
import torch_geometric
import sys
sys.path.insert(0, "../")
from v1.main import KFOLD

def load_dataset(pts,args):
    all_files = glob.glob(f"{args.graph_path}/{args.fold}/graphs_{args.numk}/*.pt")
    
    selected_files = []
    
    for i in all_files:
        for j in pts:
            if i.endswith(str(j) + ".pt"):
                graph = torch.load(i)
                selected_files.append(graph)
    return selected_files

def main(args):

    cwd = os.getcwd()
    
    def write(director,name,*string):
        string = [str(i) for i in string]
        string = " ".join(string)
        with open(os.path.join(director,name),"a") as f:
            f.write(string + "\n")
            
    store_dir = os.path.join(args.output,str(args.fold))
    print = partial(write,cwd,args.output + "/" + "log_f" + str(args.fold)) 
        
    os.makedirs(store_dir, exist_ok= True)
    
    print(args)
    

    train_patient, test_patient = KFOLD[args.fold]
    
    train_dataset = load_dataset(train_patient,args)
    test_dataset = load_dataset(test_patient, args)
    
    train_loader = torch_geometric.loader.DataLoader(
        train_dataset,
        batch_size=1,
        )
    
    
    test_loader = torch_geometric.loader.DataLoader(
        test_dataset,
        batch_size=1,
        )
    
    model = HeteroGNN(args.num_layers,args.mdim)
    CONFIG = collections.namedtuple('CONFIG', ['lr', 'logfun', 'verbose_step', 'weight_decay', 'store_dir'])
    config = CONFIG(args.lr, print, args.verbose_step, args.weight_decay,store_dir)
    
    model = TrainerModel(config, model)
    
    plt = pl.Trainer(max_epochs = args.epoch,num_nodes=1, gpus=args.gpus, val_check_interval = args.val_interval,strategy=DDPPlugin(find_unused_parameters=False),checkpoint_callback = False, logger = False)
    plt.fit(model,train_dataloaders=train_loader,val_dataloaders=test_loader)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--epoch", default = 30, type = int)
    parser.add_argument("--fold", default = 0, type = int)
    parser.add_argument("--gpus", required=True, type = int)
    parser.add_argument("--acce", default = "ddp", type = str)
    parser.add_argument("--val_interval", default = 0.8, type = float)
    parser.add_argument("--lr", required=True, type = float)
    parser.add_argument("--verbose_step", default = 10, type = int)
    parser.add_argument("--weight_decay", required=True, type = float)
    parser.add_argument("--mdim", required=True, type = int)
    parser.add_argument("--output", default = "results", type = str)
    parser.add_argument("--numk", required=True, type = int)
    parser.add_argument("--num_layers", required=True, type = int)
    parser.add_argument("--graph_path", required=True, type = str)
    
    args = parser.parse_args()
    main(args)
    
