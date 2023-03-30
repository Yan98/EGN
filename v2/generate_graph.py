import numpy as np
import torch
import os
from collections import namedtuple
import torch_geometric
import sys
sys.path.insert(0, "../")
from v1.dataset import TxPDataset
from v1.main import KFOLD
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--savename",required=True,type=str)
parser.add_argument("--size",required=True,type=int)
parser.add_argument("--numk",required=True,type=int)
parser.add_argument("--mdim",required=True,type=str)
parser.add_argument("--index_path",required=True,type=str)
parser.add_argument("--emb_path",required=True,type=str)
parser.add_argument("--data",required=True,type=str)

args = parser.parse_args()  
   

def get_edge(x):

     edge_index = torch_geometric.nn.radius_graph(
            x,
            np.sqrt(2),
            None,
            False,
            max_num_neighbors=5,
            flow="source_to_target",
            num_workers=1,
        )
     
     return edge_index


def get_cross_edge(x):
    
    l = len(x)
    source = torch.LongTensor(range(l))
    
    op = torch.cat([i[3] for i in x]).clone()
    opy = torch.cat([i[4] for i in x]).clone()

    b,n,c= op.shape
    source = torch.repeat_interleave(source, n)
    
    ops = torch.cat((op,opy),-1).view(b*n,-1)
    ops,inverse = torch.unique(ops,dim=0, return_inverse=True)
    unique_op = ops[:,:c]
    unique_opy = ops[:,c:]
    
    edge = torch.stack((source,inverse))
    return unique_op, unique_opy, edge


for fold in [0,1,2]:    
    savename = args.savename + "/" + str(fold)
    os.makedirs(savename,exist_ok=True)
    
    temp_arg = namedtuple("arg",["size","numk","mdim", "index_path", "emb_path", "data"])
    temp_arg = temp_arg(args.size, args.numk,args.mdim, args.emb_path + f"/{fold}/" + args.index_path, args.emb_path, args.data) 
    train_dataset = TxPDataset(KFOLD[fold][0] , None, None, temp_arg, train = True)
    
    
    temp_arg = namedtuple("arg",["size","numk","mdim", "index_path", "emb_path", "data"])
    temp_arg = temp_arg(args.size, args.numk,args.mdim, args.emb_path + f"/{fold}/" + args.index_path, args.emb_path, args.data)
    foldername = f"{savename}/graphs_{args.numk}"
    os.makedirs(foldername, exist_ok=True)                    
    
    for iid in range(len(KFOLD[fold][0]) + len(KFOLD[fold][1])):
        dataset = TxPDataset([iid], None, None, temp_arg,train=False)
        
        dataset.min = train_dataset.min.clone()
        dataset.max = train_dataset.max.clone()
        
        loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1
        )
        img_data = []
        for x in loader:
            pos, p, py, op, opy = x["pos"], x["p_feature"], x["count"], x["op_feature"], x["op_count"]
            img_data.append([pos, p, py, op, opy])
        
        window_edge = get_edge(torch.cat(([i[0] for i in img_data])).clone())
        
        unique_op, unique_opy, cross_edge = get_cross_edge(img_data)
        
        print(window_edge.size(), unique_op.size(), unique_opy.size(), cross_edge.size())
        
        
        data = torch_geometric.data.HeteroData()
        
        data["window"].pos = torch.cat(([i[0] for i in img_data])).clone()
        data["window"].x = torch.cat(([i[1] for i in img_data])).clone()
        data["window"].x = data["window"].x.squeeze()
        data["window"].y = torch.cat(([i[2] for i in img_data])).clone()
        
        assert len(data["window"]["pos"]) == len(data["window"]["x"]) == len(data["window"]["y"])
        
        data["example"].x = torch.cat((unique_op, unique_opy),-1)
        
        data['window', 'near', 'window'].edge_index = window_edge
        data["example", "refer", "window"].edge_index = cross_edge[[1,0]]
        
        
        edge_index = torch_geometric.nn.knn_graph(data["example"]["x"], k=3, loop=False)
        data["example", "close", "example"].edge_index = edge_index
        
        torch.save(data, f"{foldername}/{iid}.pt")
    

    
