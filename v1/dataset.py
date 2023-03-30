from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle
import pandas
import tifffile
from PIL import Image
from scanpy import read_visium 
import scanpy as sc
import pandas as pd

'''
code is based on https://github.com/bryanhe/ST-Net

'''



Breast_Cancer = [
    "1.0.0/V1_Breast_Cancer_Block_A_Section_1",
    "1.0.0/V1_Breast_Cancer_Block_A_Section_2",  
    "1.1.0/V1_Breast_Cancer_Block_A_Section_1",
    "1.1.0/V1_Breast_Cancer_Block_A_Section_2", 
    "1.2.0/Parent_Visium_Human_BreastCancer",
    "1.2.0/Targeted_Visium_Human_BreastCancer_Immunology"
    ]  

'''
(Breast_Cancer[0], Breast_Cancer[2]) and (Breast_Cancer[1], Breast_Cancer[3]) are data with 
same slide images but different gene expression, simulating noisy and contaminated experiment environments. 
If we directly use fold (Breast_Cancer[0], Breast_Cancer[1]) and fold (Breast_Cancer[4], Breast_Cancer[4]), 
we will have PCC 0.370 and PCC 0.414 for EGN and EGGN respectively. 
'''


class TxPDataset(Dataset):
    def __init__(self, breast_cancers,index_filter, transform, args, train = None):
        self.breast_cancers = breast_cancers
        self.args = args
        self.train = train
        self.index_filter = index_filter
        self.data = self.load_raw(args.data)
        self.meta_info(args.data)
        self.transform = transform
        
        keep = set(list(zip(*sorted(zip(self.mean, range(self.mean.shape[0])))[::-1][:250]))[1])
        self.filter_name = [j for i,j in  enumerate(self.gene_names) if i in keep]
        self.gene_filter = np.array([i in keep for i in range(len(self.gene_names))])
        self.max = torch.log10(torch.as_tensor(self.max[self.gene_filter],dtype=torch.float) + 1)
        self.min = torch.log10(torch.as_tensor(self.min[self.gene_filter],dtype=torch.float) + 1)
        del self.data
        mapping = []
        for i in breast_cancers:
            img,counts,coord,emb, index = self.all_data[i]
            for j in range(len(counts)):
                mapping.append([i,j])
        self.map = mapping
        
         
    def load_raw(self, data_root):
        
        
        data = []
        for idx, file in enumerate(Breast_Cancer):
            index = os.path.join(self.args.index_path, f"{idx}.npy")
            index = np.load(index)
            
            path = os.path.join(data_root,file)
            h5 = read_visium(path,count_file=f"{path.split(os.sep)[-1]}_filtered_feature_bc_matrix.h5")
            img = tifffile.imread(path + f"/{path.split(os.sep)[-1]}_image.tif")
            h5.var_names_make_unique()
            h5.var["mt"] = h5.var_names.str.startswith("MT-")
            sc.pp.calculate_qc_metrics(h5, qc_vars=["mt"], inplace=True)
            data.append([img,h5,idx,index])

        return data
    
    def meta_info(self, root):

        from tqdm import tqdm
        gene_names = set()
        for _, p, _ , _ in tqdm(self.data):
            counts = pd.DataFrame(p.X.todense(), columns=p.var_names, index=p.obs_names)
            coord = pd.DataFrame(p.obsm['spatial'], columns=['x_coord', 'y_coord'], index=p.obs_names)
            gene_names = gene_names.union(
                set(counts.columns.values)
                )
                
        gene_names = list(gene_names)
        gene_names.sort()
        
        all_data = {}
        all_gene = []
        part_gene = []
        for img,p,idx, index in tqdm(self.data):
            counts = pd.DataFrame(p.X.todense(), columns=p.var_names, index=p.obs_names)
            coord = pd.DataFrame(p.obsm['spatial'], columns=['x_coord', 'y_coord'], index=p.obs_names)
            
            missing = list(set(gene_names) - set(counts.columns.values))
            c = counts.values.astype(float)
            pad = np.zeros((c.shape[0], len(missing)))
            c = np.concatenate((c, pad), axis=1)
                
            names = np.concatenate((counts.columns.values, np.array(missing)))
            c = c[:, np.argsort(names)]
            
            emb = torch.load(f"{self.args.emb_path}/{idx}.pt",map_location=torch.device("cpu"))

            assert emb.size(0) == c.shape[0]
            
            all_data[idx] = [img,c,coord.values.astype(int),emb, index]
            
            for i in c:
                all_gene.append(i)
                if idx in self.breast_cancers:
                    part_gene.append(i)        
                
                
        all_gene = np.array(all_gene)
        part_gene = np.array(part_gene)
        print(all_gene.shape, part_gene.shape)
        
        self.mean = np.mean(all_gene, 0)
        self.max  = np.max(part_gene,0)
        self.min  = np.min(part_gene,0)
        self.gene_names = gene_names
        self.all_data = all_data
        
    def retrive_similer(self, index):
        numk = self.args.numk
        index = np.array([[i,j,k] for i,j,k in sorted(index, key = lambda x : float(x[0]))]) 
        index = index[-numk:]
    
        op_emb = []
        op_counts = []
        for _, op_name, op_n in index:
            op_name = int(op_name)
            op_n = int(op_n)
            op_emb.append(self.all_data[op_n][3][op_name])
            op_count = torch.as_tensor(self.all_data[op_n][1][op_name],dtype=torch.float)
            op_count = torch.log10(op_count[self.gene_filter] + 1)
            op_count = (op_count - self.min) / (self.max - self.min + 1e-8)
            op_counts.append(op_count)
    
        return torch.stack(op_emb).view(numk,-1), torch.stack(op_counts).view(numk,250)    
        
    def generate(self, idx):

        idx = self.map[idx]
        img,counts,coord,emb, index = self.all_data[idx[0]]
        counts, coord, emb, index = counts[idx[1]],coord[idx[1]],emb[idx[1]], index[idx[1]]
        
        emb = emb.unsqueeze(0)
        
        x,y = coord
        window = self.args.size
        pos = [x//window, y//window]
        
        op_emb, op_counts = self.retrive_similer(index)
        
        img = img[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]
        
        if self.transform != None:
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.as_tensor(img,dtype=torch.float).permute(2,0,1) / 255   
        
        counts = torch.log10(torch.as_tensor(counts[self.gene_filter],dtype = torch.float) + 1)
        counts = (counts - self.min) / (self.max - self.min + 1e-8)
        
        return {
            "img" : img,
             "count" : counts,
             "p_feature": emb,
             "op_count": op_counts,
             "op_feature":op_emb,
             "pos": torch.LongTensor(pos),
            }
    
    def  __getitem__(self, index):
        return self.generate(index)
        
    def __len__(self):
        return len(self.map)

