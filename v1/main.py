import argparse
import os
from torch.utils import data
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from functools import partial
import torch
import collections
from train import TrainerModel
from sklearn.model_selection import KFold
import torchvision
from pytorch_lightning.plugins import DDPPlugin

cudnn.benchmark = True

XFOLD = [0,1,2,3,4,5]
skf = KFold(n_splits=3,shuffle= True, random_state = 12345)
KFOLD = []
for x in skf.split(XFOLD):
    KFOLD.append(x)

mean = [0.5476, 0.5218, 0.6881]
std  = [0.2461, 0.2101, 0.1649] 

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
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
        ])

    train_dataset = TxPDataset(KFOLD[args.fold][0] , KFOLD[args.fold][1], transform, args, train = True)
    test_dataset = TxPDataset(KFOLD[args.fold][1]  , KFOLD[args.fold][1], transform, args, train=False)
    test_dataset.min = train_dataset.min.clone()
    test_dataset.max = train_dataset.max.clone()
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        num_workers = args.workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last = True
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch,
        num_workers = args.workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last = True
    )
    
    filter_name = train_dataset.filter_name
    model = EGN(
        image_size = args.size,
        dim = args.dim,
        depth = args.depth,
        heads = args.heads,
        mlp_dim = args.mlp_dim,
        bhead = args.bhead,
        bdim = args.bdim,
        bfre = args.bfre,
        mdim=args.mdim,
        player=args.player,
        linear_projection=args.linear_projection,
        )
    CONFIG = collections.namedtuple('CONFIG', ['lr', 'logfun', 'verbose_step', 'weight_decay', 'store_dir', 'filter_name'])
    config = CONFIG(args.lr, print, args.verbose_step, args.weight_decay,store_dir,filter_name)
        
    if args.checkpoints != None:
        model.load_state_dict(torch.load(args.checkpoints,map_location = torch.device("cpu")))
    
    model = TrainerModel(config, model)
    
    plt = pl.Trainer(max_epochs = args.epoch,num_nodes=args.num_nodes, gpus=args.gpus, val_check_interval = args.val_interval,strategy=DDPPlugin(find_unused_parameters=False),checkpoint_callback = False, logger = False)
    plt.fit(model,train_dataloaders=train_loader,val_dataloaders=test_loader)
    
    del train_dataset
    del test_dataset
    del train_loader
    del test_loader

if __name__ == "__main__":
    
    from dataset import TxPDataset
    from egn import EGN

    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--epoch", default = 50, type = int)
    parser.add_argument("--fold", default = 0, type = int)
    parser.add_argument("--gpus", required=True, type = int)
    parser.add_argument("--acce", default = "ddp", type = str)
    parser.add_argument("--val_interval", default = 0.8, type = float)
    parser.add_argument("--lr", required=True, type = float)
    parser.add_argument("--verbose_step", default = 10, type = int)
    parser.add_argument("--weight_decay", required=True, type = float)
    parser.add_argument("--dim", required=True, type = int)
    parser.add_argument("--heads", required=True, type = int)
    parser.add_argument("--mlp_dim", required=True, type = int)
    parser.add_argument("--depth", required=True,type = int)
    parser.add_argument("--batch", required=True, type = int)
    parser.add_argument("--workers", required=True, type = int)
    parser.add_argument("--checkpoints", default = None, type = str)
    parser.add_argument("--output", default = "results", type = str)
    parser.add_argument("--num_nodes", required=True, type = int)
    parser.add_argument("--size", required=True, type = int)
    parser.add_argument("--bhead", required=True, type = int)
    parser.add_argument("--bdim", required=True, type = int)
    parser.add_argument("--bfre", required=True, type = int)
    parser.add_argument("--data", required=True, type = str)
    parser.add_argument("--index_path", required=True, type = str)
    parser.add_argument("--emb_path", required=True, type = str)
    parser.add_argument("--mdim", required=True, type = int)
    parser.add_argument("--player", required=True, type = int)
    parser.add_argument("--linear_projection", required=True, type = bool)
    parser.add_argument("--numk", required=True, type = int)
    
    args = parser.parse_args()
    main(args)
    
