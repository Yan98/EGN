import os
import torch
import tifffile
import heapq
import torchvision
import numpy as np
from tqdm import tqdm
import argparse
from scanpy import read_visium
import scanpy as sc
import pandas as pd
from joblib import Parallel, delayed
import sys
sys.path.insert(0, "../")
from v1.dataset import Breast_Cancer
from v1.main import KFOLD 


parser = argparse.ArgumentParser()
parser.add_argument("--index_dir",required=True,type=str)
parser.add_argument("--save_dir",required=True,type=str)
parser.add_argument("--TORCH_HOME",required=True,type=str)
parser.add_argument("--data",required=True,type=str)
parser.add_argument("--fold",required=True,type=int)

args = parser.parse_args()  
   
index_dir = args.index_dir
save_dir = args.save_dir
reference=KFOLD[args.fold][0]

os.makedirs(os.path.join(save_dir, str(args.fold),index_dir),exist_ok=True)
os.environ['TORCH_HOME'] = args.TORCH_HOME
encoder = torchvision.models.resnet18(True)
features = encoder.fc.in_features
modules=list(encoder.children())[:-1]
encoder=torch.nn.Sequential(*modules)
for p in encoder.parameters():
    p.requires_grad = False
encoder=encoder.cuda()
encoder.eval()

num_cores = 12
batch_size = 64
window = 256      

root = args.data

data = []
for file in Breast_Cancer:
    path = os.path.join(root,file)
    h5 = read_visium(path,count_file=f"{path.split(os.sep)[-1]}_filtered_feature_bc_matrix.h5")
    img = tifffile.imread(path + f"/{path.split(os.sep)[-1]}_image.tif")
    h5.var_names_make_unique()
    h5.var["mt"] = h5.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(h5, qc_vars=["mt"], inplace=True)
    
    coord = pd.DataFrame(h5.obsm['spatial'], columns=['x_coord', 'y_coord'], index=h5.obs_names)
    data.append([img,coord.values.astype(int)])
    print(coord.values.astype(int).shape)
    
mapping = []
for i,(img,coord) in enumerate(data):
    mappings = []
    for j in range(len(coord)):
        mappings.append(j)
    mapping.append(mappings)

            
def generate():
                    
    def get_slide_gene(idx,):
        
        img,coord = data[idx[0]]
        coord = coord[idx[1]]
        
        x,y = coord
        img = img[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]
        
        
        code = idx[1]
        img = torch.as_tensor(img,dtype=torch.float).permute(2,0,1) / 255   
        img = img.cuda()           
        return (img - 0.5) / 0.5 , code
    
    def extract(imgs):
        img = torch.stack(imgs)
        return encoder(img).view(-1,features)
    
    for i, k in tqdm(enumerate(mapping)):
        batch_img, codes = [], []
        img_embedding = []
        for j in k:
            img, code = get_slide_gene([i,j])
            batch_img.append(img)
            codes.append(code)
            
            if len(batch_img) == batch_size:
                img_embedding += [extract(batch_img)]
                batch_img =  []
            
        if len(batch_img) != 0:
            img_embedding += [extract(batch_img)]
        
        img_embedding = torch.cat(img_embedding).contiguous()
        
        assert (np.array(codes) == np.sort(codes)).all()
        
        assert  img_embedding.size(0) == len(codes)
        
        assert img_embedding.size(1) == features
        print(img_embedding.size())
        torch.save(img_embedding, f"{save_dir}/{i}.pt")


def create_search_index():
    
    class Queue:
        def __init__(self, max_size = 2):
            self.max_size = max_size
            self.list = []

        def add(self, item):

            heapq.heappush(self.list, item)

            while len(self.list) > self.max_size:
                heapq.heappop(self.list)
    
        def __repr__(self):
            return str(self.list)

    for i, _ in tqdm(enumerate(mapping)):
        p = f"{save_dir}/{i}.pt"
        p = torch.load(p).cuda()
        
        Q = [Queue(max_size=128) for _ in range(p.size(0))]
           
        for op_i, _ in enumerate(mapping):
            if op_i == i or not op_i in reference:
                continue

            op = torch.load(f"{save_dir}/{op_i}.pt").cuda()
            dist = torch.cdist(p.unsqueeze(0),op.unsqueeze(0),p = 1).squeeze(0)
            topk = min(len(dist),100)
            knn = dist.topk(topk, dim = 1, largest=False)

            q_values = knn.values.cpu().numpy()
            q_infos =  knn.indices.cpu().numpy() 

            def add(q_value,q_info, myQ):
                for idx in range(len(q_value)):
                    myQ.add((-q_value[idx],q_info[idx],op_i))
                return myQ

            Q = Parallel(n_jobs=num_cores)(delayed(add)(q_values[f],q_infos[f],Q[f]) for f in range(q_values.shape[0]))
        np.save(f"{save_dir}/{args.fold}/{index_dir}/{i}.npy", [myq.list for myq in Q] )
    
if args.fold==0:
    generate()
create_search_index()
        
        
                